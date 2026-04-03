import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import sys
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))
from path_utils import MODELS

# Re-define VAE internally for standalone inference if needed
class VAE(nn.Module):
    def __init__(self, input_dim=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2_mu = nn.Linear(h_dim, z_dim)
        self.fc2_log_var = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2_mu(h), self.fc2_log_var(h)

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

def load_vae_model():
    model = VAE()
    model_path = MODELS / "vae_mnist.pkl"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def generate_samples(model, n=64):
    with torch.no_grad():
        z = torch.randn(n, 20)
        samples = model.decode(z).view(-1, 28, 28).cpu().numpy()
    
    # Create grid
    grid_size = int(np.sqrt(n))
    img_grid = np.zeros((grid_size * 28, grid_size * 28))
    for i in range(n):
        r, c = i // grid_size, i % grid_size
        img_grid[r*28:(r+1)*28, c*28:(c+1)*28] = samples[i]
    
    return Image.fromarray((img_grid * 255).astype(np.uint8))

def interpolate_digits(model, z1, z2, steps=10):
    imgs = []
    zs = []
    for t in np.linspace(0, 1, steps):
        z_interp = (1 - t) * z1 + t * z2
        zs.append(z_interp.squeeze().cpu().numpy())
        with torch.no_grad():
            img = model.decode(z_interp).view(28, 28).cpu().numpy()
        imgs.append(Image.fromarray((img * 255).astype(np.uint8)))
    return imgs, np.array(zs)

def get_latent_vector(model, seed):
    torch.manual_seed(seed)
    z = torch.randn(1, 20)
    return z.squeeze().cpu().numpy()

def get_training_history():
    history_path = MODELS / "history.json"
    if history_path.exists():
        with open(history_path, "r") as f:
            return json.load(f)
    return None
