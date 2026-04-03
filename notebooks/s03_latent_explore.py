import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils
from sklearn.manifold import TSNE
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from path_utils import MODELS, CHARTS
from notebooks.s02_train import VAE

def stop_8_latent_viz(model, test_loader, device="cpu"):
    print("\n[STOP 8] Visualizing Latent Space (t-SNE Projection)...")
    model.eval()
    z_list = []
    labels_list = []
    
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            mu, log_var = model.encode(data.view(-1, 784))
            z_list.append(mu.cpu().numpy())
            labels_list.append(labels.numpy())
            if i > 20: break # Use only first 20 batches for speed
            
    z_all = np.concatenate(z_list)
    labels_all = np.concatenate(labels_list)
    
    tsne = TSNE(n_components=2, perplexity=30, max_iter=300, random_state=42)
    z_2d = tsne.fit_transform(z_all)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels_all, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter)
    plt.title("VAE Latent Space t-SNE Projection (MNIST)")
    plt.savefig(CHARTS / "latent_space_mnist.png")
    print(f"Latent viz saved to {CHARTS}/latent_space_mnist.png")

def stop_9_interpolation(model, device="cpu"):
    print("\n[STOP 9] Latent Space Interpolation (Digit Morphing)...")
    model.eval()
    
    # Random seeds for mu
    z1 = torch.randn(1, 20).to(device)
    z2 = torch.randn(1, 20).to(device)
    
    steps = 10
    imgs = []
    for t in np.linspace(0, 1, steps):
        z_interp = (1 - t) * z1 + t * z2
        with torch.no_grad():
            img = model.decode(z_interp)
        imgs.append(img.view(1, 28, 28))
    
    # Display in a row
    grid = utils.make_grid(torch.stack(imgs), nrow=steps)
    plt.figure(figsize=(15, 3))
    plt.axis("off")
    plt.title("Digit Morphing Example")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.savefig(CHARTS / "interpolation.png")
    print(f"Interpolation grid saved to {CHARTS}/interpolation.png")

def stop_10_sampling(model, n=64, device="cpu"):
    print("\n[STOP 10] Generating New Digits from Prior N(0, 1)...")
    model.eval()
    
    with torch.no_grad():
        z = torch.randn(n, 20).to(device)
        samples = model.decode(z)
        
    grid = utils.make_grid(samples.view(-1, 1, 28, 28), nrow=8)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(f"Genereated Samples (n={n})")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.savefig(CHARTS / "generated_samples.png")
    print(f"Samples grid saved to {CHARTS}/generated_samples.png")

def run_exploration(model=None, test_loader=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load model if not provided
    if model is None:
        model = VAE().to(device)
        model.load_state_dict(torch.load(MODELS / "vae_mnist.pkl", map_location=device))
        
    stop_8_latent_viz(model, test_loader, device=device)
    stop_9_interpolation(model, device=device)
    stop_10_sampling(model, n=64, device=device)
    
    print("\n[STOP 13] VAE Latent Exploration Complete.")

if __name__ == "__main__":
    run_exploration()
