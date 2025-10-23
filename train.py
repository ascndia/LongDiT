import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import os
from dit import DiT  # Ensure you have the DiT class implemented
# --- Parameters ---
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
IMG_SIZE = 28
CHANNELS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Anisotropic patches for a 28x28 image
# (1, 2) -> 28x14 grid -> L=392
# (2, 1) -> 14x28 grid -> L=392
# Total L = 784 patches
PATCH_STRATEGY_H = (1, 2)
PATCH_STRATEGY_V = (2, 1)

SAMPLE_DIR = "samples"
os.makedirs(SAMPLE_DIR, exist_ok=True)

# --- Sampling Function (ODE Solver) ---
@torch.no_grad()
def sample(model, device, steps=100, batch_size=16):
    """
    Sample from the model using a simple Euler ODE solver.
    """
    print("Sampling images...")
    model.eval()

    # Start from pure noise (t=0)
    x = torch.randn((batch_size, CHANNELS, IMG_SIZE, IMG_SIZE), device=device)

    dt = 1.0 / steps
    time_steps = torch.linspace(0, 1, steps, device=device)

    for i in tqdm(range(steps - 1)):
        t = time_steps[i]

        # Get the predicted vector field v(x_t, t)
        v_pred = model(x, t.repeat(batch_size))

        # Euler step: x_{t+dt} = x_t + v(x_t, t) * dt
        x = x + v_pred * dt

    model.train()

    # De-normalize from [-1, 1] to [0, 1]
    x = (x.clamp(-1, 1) + 1) / 2
    return x

# --- Main Training Script ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    # 2. Setup Model
    # We must pass all the required arguments to your DiT class
    model = DiT(
        in_dim=IMG_SIZE,
        channels=CHANNELS,
        depth=12,
        num_heads=4,
        dim_head=64,
        patch_strategy_h=PATCH_STRATEGY_H,
        patch_strategy_v=PATCH_STRATEGY_V,
    ).to(DEVICE)

    # Print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # 3. Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch+1}/{EPOCHS} ---")

        for i, (x1, _) in enumerate(tqdm(loader)):

            # --- Flow Matching Logic ---

            # 1. Get data and noise
            x1 = x1.to(DEVICE)                       # Data: x_1 ~ P_data
            x0 = torch.randn_like(x1)                # Noise: x_0 ~ N(0, I)

            # 2. Sample time t
            # t ~ U(0, 1)
            t = torch.rand(x1.shape[0], device=DEVICE)
            t_broadcast = t.view(-1, 1, 1, 1)

            # 3. Create interpolated path: x_t = (1-t)x_0 + t*x_1
            xt = (1 - t_broadcast) * x0 + t_broadcast * x1

            # 4. Define target vector field: v_t = x_1 - x_0
            vt = x1 - x0

            # --- Model Forward Pass ---
            optimizer.zero_grad()

            # Get predicted vector field: v_pred = model(x_t, t)
            # The DiT class expects the time/sigma embed as a 1D tensor
            vt_pred = model(xt, t)

            # 5. Calculate Loss
            loss = nn.MSELoss()(vt_pred, vt)

            # 6. Backprop
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                print(f"Epoch {epoch+1}, Step {i}: Loss = {loss.item():.4f}")

        # --- End of Epoch ---
        # Save a sample image
        samples = sample(model, DEVICE, steps=100, batch_size=16)
        save_image(samples, os.path.join(SAMPLE_DIR, f"epoch_{epoch+1:03d}.png"), nrow=4)

        # Save model checkpoint
        torch.save(model.state_dict(), f"dit_anisotropic_epoch_{epoch+1}.pth")

    print("Training complete.")