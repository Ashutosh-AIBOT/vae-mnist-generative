import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from notebooks.s01_theory import run_theory
from notebooks.s02_train import run_vae_training, stop_4_dataloaders
from notebooks.s03_latent_explore import run_exploration

def run_full_vae_pipeline(epochs=10, auto_mode=False):
    print("\n" + "="*60)
    print("      VARIATIONAL AUTOENCODER (VAE) MASTER PIPELINE       ")
    print("="*60 + "\n")
    
    # 1. Theory Module
    run_theory(auto_mode=auto_mode)
    
    # 2. Training Module
    print(f"\n[PIPELINE] Starting VAE Training ({epochs} epochs)...")
    model, history = run_vae_training(epochs=epochs)
    
    # 3. Exploration Module
    _, test_loader = stop_4_dataloaders()
    print("\n[PIPELINE] Starting Latent Space Exploration & Generation...")
    run_exploration(model=model, test_loader=test_loader)
    
    print("\n" + "="*60)
    print("           VAE PIPELINE EXECUTION COMPLETE            ")
    print("="*60 + "\n")
    print("Next Step: Launch Dashboard via 'streamlit run app.py'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE Pipeline Runner")
    parser.add_argument("--auto", action="store_true", help="Run in non-interactive mode")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    
    args = parser.parse_args()
    run_full_vae_pipeline(epochs=args.epochs, auto_mode=args.auto)
