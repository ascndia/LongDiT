import torch
import numpy as np
from torch import nn, functional as F
from einops import rearrange, repeat
# Assuming these are in your .model file
## Diffusion transformer

from favor import Attention
# class Attention(nn.Module):
#     def __init__(self, dim_head, num_heads=8, qkv_bias=False):
#         super().__init__()
#         self.num_heads = num_heads
#         self.dim_head = dim_head
#         dim = dim_head * num_heads
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.proj = nn.Linear(dim, dim)

#     def forward(self, x):
#         # (B, N, D) -> (B, N, D)
#         # N = H * W / patch_size**2, D = num_heads * dim_head
#         q, k, v = rearrange(self.qkv(x), 'b n (qkv h k) -> qkv b h n k',
#                             h=self.num_heads, k=self.dim_head)
#         x = rearrange(F.scaled_dot_product_attention(q, k, v),
#                       'b h n k -> b n (h k)')
#         return self.proj(x)
    
## MODIFIED: Now accepts patch_h and patch_w
class PatchEmbed(nn.Module):
    def __init__(self, patch_h=16, patch_w=16, channels=3, embed_dim=768, bias=True):
        super().__init__()
        self.proj = nn.Conv2d(channels, embed_dim, stride=(patch_h, patch_w), kernel_size=(patch_h, patch_w), bias=bias)
        self.init()

    def init(self): # Init like nn.Linear
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        return rearrange(self.proj(x), 'b c h w -> b (h w) c')

class Modulation(nn.Module):
    def __init__(self, dim, n):
        super().__init__()
        self.n = n
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(dim, n * dim, bias=True))
        nn.init.constant_(self.proj[-1].weight, 0)
        nn.init.constant_(self.proj[-1].bias, 0)

    def forward(self, y):
        return [m.unsqueeze(1) for m in self.proj(y).chunk(self.n, dim=1)]

class ModulatedLayerNorm(nn.LayerNorm):
    def __init__(self, dim, **kwargs):
        super().__init__(dim, **kwargs)
        self.modulation = Modulation(dim, 2)
    def forward(self, x, y):
        scale, shift = self.modulation(y)
        return super().forward(x) * (1 + scale) + shift

class DiTBlock(nn.Module):
    def __init__(self, dim_head, num_heads, mlp_ratio=4.0):
        super().__init__()
        dim = dim_head * num_heads
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = ModulatedLayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim_head, num_heads=num_heads, qkv_bias=True)
        self.norm2 = ModulatedLayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim, bias=True),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_hidden_dim, dim, bias=True),
        )
        self.scale_modulation = Modulation(dim, 2)

    def forward(self, x, y):
        # (B, N, D), (B, D) -> (B, N, D)
        # N = H * W / patch_size**2, D = num_heads * dim_head
        gate_msa, gate_mlp = self.scale_modulation(y)
        x = x + gate_msa * self.attn(self.norm1(x, y))
        x = x + gate_mlp * self.mlp(self.norm2(x, y))
        return x

## NEW: A cleaner 2D Positional Embedding function for rectangular grids
def get_2d_pos_embed(grid_h, grid_w, dim, N=10000):
    """
    Generates 2D Sin/Cos Positional Embedding for a rectangular grid.
    """
    assert dim % 4 == 0, 'Embedding dimension must be multiple of 4!'
    
    # 1. Get frequencies
    omega = 1 / N**np.linspace(0, 1, dim // 4, endpoint=False) # [dim/4]

    # 2. Get 1D position encodings
    freqs_h = np.outer(np.arange(grid_h), omega) # [grid_h, dim/4]
    freqs_w = np.outer(np.arange(grid_w), omega) # [grid_w, dim/4]
    
    # 3. Stack sin/cos
    embed_h = np.stack([np.sin(freqs_h), np.cos(freqs_h)], axis=-1) # [grid_h, dim/4, 2]
    embed_w = np.stack([np.sin(freqs_w), np.cos(freqs_w)], axis=-1) # [grid_w, dim/4, 2]

    # 4. Reshape and tile
    embed_h = rearrange(embed_h, 'h d b -> h (d b)') # [grid_h, dim/2]
    embed_w = rearrange(embed_w, 'w d b -> w (d b)') # [grid_w, dim/2]
    
    embed_h = repeat(embed_h, 'h d -> (h w) d', w=grid_w) # [L, dim/2]
    embed_w = repeat(embed_w, 'w d -> (h w) d', h=grid_h) # [L, dim/2]
    
    # 5. Concatenate
    embeds_2d = np.concatenate([embed_h, embed_w], axis=1) # [L, dim]
    
    return nn.Parameter(torch.tensor(embeds_2d).float().unsqueeze(0), # [1, L, dim]
                        requires_grad=False)

class SigmaEmbedderSinCos(nn.Module):
    def __init__(self, hidden_size, scaling_factor=0.5, log_scale=True):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.log_scale = log_scale
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

## MODIFIED: This is now the AnisotropicDiT
class DiT(nn.Module):
    def __init__(self, in_dim=32, channels=3, depth=12,
                 dim_head=64, num_heads=6, mlp_ratio=4.0,
                 sig_embed=None, cond_embed=None,
                 ## NEW: Define the patch strategies here
                 patch_strategy_h=(1, 2), # (patch_h, patch_w) for horizontal
                 patch_strategy_v=(2, 1)  # (patch_h, patch_w) for vertical
                ):
        super().__init__()
        self.in_dim = in_dim
        self.channels = channels
        self.input_dims = (channels, in_dim, in_dim)

        dim = dim_head * num_heads

        # --- 1. Horizontal (1x2) Stream Setup ---
        self.patch_h_h, self.patch_w_h = patch_strategy_h
        self.grid_h_h = in_dim // self.patch_h_h
        self.grid_w_h = in_dim // self.patch_w_h
        self.L_h = self.grid_h_h * self.grid_w_h
        self.x_embed_h = PatchEmbed(self.patch_h_h, self.patch_w_h, channels, dim, bias=True)
        self.pos_embed_h = get_2d_pos_embed(self.grid_h_h, self.grid_w_h, dim)
        self.type_embed_h = nn.Parameter(torch.zeros(1, 1, dim))

        # --- 2. Vertical (2x1) Stream Setup ---
        self.patch_h_v, self.patch_w_v = patch_strategy_v
        self.grid_h_v = in_dim // self.patch_h_v
        self.grid_w_v = in_dim // self.patch_w_v
        self.L_v = self.grid_h_v * self.grid_w_v
        self.x_embed_v = PatchEmbed(self.patch_h_v, self.patch_w_v, channels, dim, bias=True)
        self.pos_embed_v = get_2d_pos_embed(self.grid_h_v, self.grid_w_v, dim)
        self.type_embed_v = nn.Parameter(torch.zeros(1, 1, dim))

        # --- 3. Core Transformer ---
        self.sig_embed = sig_embed or SigmaEmbedderSinCos(dim)
        self.cond_embed = cond_embed
        self.blocks = nn.ModuleList([
            DiTBlock(dim_head, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        # --- 4. Decoder and Fusion ---
        self.final_norm = ModulatedLayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # Two separate final layers
        self.final_linear_h = nn.Linear(dim, (self.patch_h_h * self.patch_w_h) * channels)
        self.final_linear_v = nn.Linear(dim, (self.patch_h_v * self.patch_w_v) * channels)
        
        # Learnable fusion layer
        self.fusion_layer = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, bias=True)
        
        self.init()

    def init(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize sigma embedding MLP:
        nn.init.normal_(self.sig_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.sig_embed.mlp[2].weight, std=0.02)
        
        ## NEW: Initialize type embeddings
        nn.init.normal_(self.type_embed_h, std=0.02)
        nn.init.normal_(self.type_embed_v, std=0.02)

        # Zero-out output layers:
        ## MODIFIED: Initialize all output layers to zero
        nn.init.constant_(self.final_linear_h.weight, 0)
        nn.init.constant_(self.final_linear_h.bias, 0)
        nn.init.constant_(self.final_linear_v.weight, 0)
        nn.init.constant_(self.final_linear_v.bias, 0)
        nn.init.constant_(self.fusion_layer.weight, 0)
        nn.init.constant_(self.fusion_layer.bias, 0)

    ## NEW: Unpatchify for Horizontal Stream
    def unpatchify_h(self, x):
        # (B, L_h, psh*psw*c) -> (B, c, H, W)
        return rearrange(x, 'b (ph pw) (psh psw c) -> b c (ph psh) (pw psw)',
                         ph=self.grid_h_h, pw=self.grid_w_h,
                         psh=self.patch_h_h, psw=self.patch_w_h,
                         c=self.channels)

    ## NEW: Unpatchify for Vertical Stream
    def unpatchify_v(self, x):
        # (B, L_v, psh*psw*c) -> (B, c, H, W)
        return rearrange(x, 'b (ph pw) (psh psw c) -> b c (ph psh) (pw psw)',
                         ph=self.grid_h_v, pw=self.grid_w_v,
                         psh=self.patch_h_v, psw=self.patch_w_v,
                         c=self.channels)

    def forward(self, x, sigma, cond=None):
        # x: (B, C, H, W), sigma: Union[(B, 1, 1, 1), ()], cond: (B, *)
        # returns: (B, C, H, W)
        
        # --- 1. Patchify and Embed ---
        # (B, C, H, W) -> (B, L_h, D)
        x_h = self.x_embed_h(x) + self.pos_embed_h + self.type_embed_h
        # (B, C, H, W) -> (B, L_v, D)
        x_v = self.x_embed_v(x) + self.pos_embed_v + self.type_embed_v
        
        # --- 2. Concatenate ---
        # (B, L_h + L_v, D)
        x = torch.cat([x_h, x_v], dim=1)
        
        # --- 3. Run Transformer ---
        # Get conditioning
        y = self.sig_embed(x.shape[0], sigma.squeeze()) # (B, D)
        if self.cond_embed is not None:
            assert cond is not None and x.shape[0] == cond.shape[0], \
                'Conditioning must have same batches as x!'
            y += self.cond_embed(cond)                   # (B, D)
        
        # Apply blocks
        x = self.blocks(x, y)                           # (B, L_h + L_v, D)
        
        # --- 4. Decode ---
        x = self.final_norm(x, y)                       # (B, L_h + L_v, D)
        
        # Split the sequence back
        x_h, x_v = torch.split(x, [self.L_h, self.L_v], dim=1) # (B, L_h, D), (B, L_v, D)
        
        # Apply separate final layers
        x_h = self.final_linear_h(x_h) # (B, L_h, psh*psw*c)
        x_v = self.final_linear_v(x_v) # (B, L_v, psh*psw*c)
        
        # Unpatchify
        img_h = self.unpatchify_h(x_h) # (B, C, H, W)
        img_v = self.unpatchify_v(x_v) # (B, C, H, W)
        
        # --- 5. Fuse ---
        img_fused = torch.cat([img_h, img_v], dim=1) # (B, 2*C, H, W)
        img_out = self.fusion_layer(img_fused)       # (B, C, H, W)
        
        return img_out