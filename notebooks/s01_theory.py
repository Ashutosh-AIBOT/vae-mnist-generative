import torch
import numpy as np
import time

def print_stop(stop_num, title, explanation, auto_mode=False):
    print(f"\n{'='*60}")
    print(f"STOP {stop_num}: {title}")
    print(f"{'='*60}")
    print(f"\nEXPLANATION:\n{explanation}")
    print(f"\n{'='*60}\n")
    if not auto_mode:
        input("Press Enter to continue to the next STOP...")

def stop_1_vae_vs_ae(auto_mode=False):
    explanation = (
        "The core difference between an Autoencoder (AE) and a Variational Autoencoder (VAE):\n"
        "1. AE: Maps input to a SINGLE point in latent space (Deterministic).\n"
        "2. VAE: Maps input to a DISTRIBUTION (Mean and Variance).\n\n"
        "Why VAE? Standard AEs have 'voids' in latent space where the decoder hasn't learned anything.\n"
        "VAEs force the latent space to be continuous and structured, enabling REAL generation."
    )
    print_stop(1, "AE vs VAE Concept", explanation, auto_mode)

def stop_2_reparameterization(auto_mode=False):
    explanation = (
        "THE REPARAMETERIZATION TRICK:\n"
        "In a VAE, we need to sample z ~ N(mu, sigma^2). However, 'sampling' is non-differentiable.\n"
        "We can't backpropagate through a random node.\n\n"
        "FIX: We express z as z = mu + sigma * epsilon, where epsilon ~ N(0, 1).\n"
        "Now, the randomness is in epsilon (an input), and z is a deterministic function of mu and sigma.\n"
        "Gradients can now flow back to the encoder!"
    )
    print_stop(2, "Reparameterization Trick", explanation, auto_mode)

def stop_3_elbo_and_kl(auto_mode=False):
    explanation = (
        "VAEs are trained using the ELBO (Evidence Lower Bound) loss:\n"
        "Total Loss = Reconstruction Loss + KL Divergence\n\n"
        "1. Reconstruction Loss: Ensures output looks like input (BCE).\n"
        "2. KL Divergence: Regularizes space by pushing μ toward 0 and σ toward 1.\n"
        "   Formula: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))"
    )
    print_stop(3, "ELBO & KL Divergence Strategy", explanation, auto_mode)

def run_theory(auto_mode=False):
    print("Initializing VAE Theory Module...")
    stop_1_vae_vs_ae(auto_mode)
    stop_2_reparameterization(auto_mode)
    stop_3_elbo_and_kl(auto_mode)
    print("\n[THEORY MODULE COMPLETE]")

if __name__ == "__main__":
    run_theory(auto_mode=False)
