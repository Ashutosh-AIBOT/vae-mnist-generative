import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from path_utils import DATA_DIR, MODELS

class VAE(nn.Module):
    def __init__(self, input_dim=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        
        # Encoder: 784 -> 400 -> (mu=20, log_var=20)
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2_mu = nn.Linear(h_dim, z_dim)
        self.fc2_log_var = nn.Linear(h_dim, z_dim)
        
        # Decoder: 20 -> 400 -> 784
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2_mu(h), self.fc2_log_var(h)

    def reparameterize(self, mu, log_var):
        # Reparameterization Trick: z = mu + std * epsilon
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h)) # Pixels in [0, 1]

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def vae_loss(recon_x, x, mu, log_var, beta=1.0):
    # BCE Reconstruction Loss (Sum over all pixels)
    recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL Divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + (beta * kl_loss), recon_loss, kl_loss

def stop_4_dataloaders(batch_size=128):
    print("\n[STOP 4] Loading MNIST Dataset...")
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Dataset Size: {len(train_dataset)} | Batch: {batch_size}")
    return train_loader, test_loader

def run_vae_training(epochs=50, batch_size=128, device="cuda" if torch.cuda.is_available() else "cpu"):
    train_loader, test_loader = stop_4_dataloaders(batch_size)
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"\n[STOP 5-7] Starting VAE Training Loop on {device}...")
    
    history = {"train_loss": [], "recon_loss": [], "kl_loss": []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        r_loss = 0
        k_loss = 0
        
        # KL Annealing Logic: Linearly increase beta from 0 to 1 over first 10 epochs
        beta = min(1.0, epoch / 10.0)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] (Beta={beta:.2f})")
        for i, (data, _) in enumerate(progress_bar):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = model(data)
            loss, recon, kl = vae_loss(recon_batch, data, mu, log_var, beta=beta)
            
            loss.backward()
            train_loss += loss.item()
            r_loss += recon.item()
            k_loss += kl.item()
            optimizer.step()
            
            if i % 100 == 0:
                progress_bar.set_postfix({"Recon": f"{recon.item()/len(data):.2f}", "KL": f"{kl.item()/len(data):.2f}"})
        
        avg_loss = train_loss / len(train_loader.dataset)
        history["train_loss"].append(avg_loss)
        history["recon_loss"].append(r_loss / len(train_loader.dataset))
        history["kl_loss"].append(k_loss / len(train_loader.dataset))
        
        print(f"====> Epoch: {epoch+1} Average loss: {avg_loss:.4f} (Beta={beta:.2f})")

    # Save Checkpoint
    torch.save(model.state_dict(), MODELS / "vae_mnist.pkl")
    # Save History
    import json
    with open(MODELS / "history.json", "w") as f:
        json.dump(history, f)
    
    print(f"\n[TRAINING COMPLETE] Model saved to {MODELS}/vae_mnist.pkl")
    return model, history

if __name__ == "__main__":
    run_vae_training(epochs=1) # Fast run for testing
