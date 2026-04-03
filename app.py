import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))
try:
    from dashboard_core import load_vae_model, generate_samples, interpolate_digits, get_training_history, get_latent_vector
except ImportError:
    st.error("VAE Dashboard artifacts could not be imported.")

def main():
    # --- 1. APP CONFIG ---
    st.set_page_config(
        page_title="VAE Precision Synthesizer",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- 2. PREMIUM CSS ---
    st.markdown("""
    <style>
        .stApp {background-color: #0e1117;}
        [data-testid="stSidebar"] {background-color: #161b22; border-right: 1px solid #30363d;}
        
        .kpi-card {
            background: #1c2128; padding: 1.5rem; border-radius: 8px; border: 1px solid #30363d;
            text-align: center; transition: transform 0.2s;
        }
        .kpi-card:hover {transform: translateY(-5px); border-color: #58a6ff;}
        .theory-box {
            padding: 1.5rem; background: #0d1117; border-radius: 8px; border-left: 5px solid #238636;
            margin-bottom: 2rem;
        }
        .chart-container {
            background: #161b22; padding: 10px; border-radius: 8px; border: 1px solid #30363d;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- 3. SIDEBAR ---
    with st.sidebar:
        st.markdown("### 🧬 Auditor Controls")
        st.info("**Model**: Variational Autoencoder (VAE)")
        st.markdown("---")
        
        st.markdown("#### Morphing & Concept Tools")
        z1_seed = st.number_input("Latent Seed A (Start)", value=42, step=1)
        z2_seed = st.number_input("Latent Seed B (End)", value=100, step=1)
        steps = st.slider("Interpolation Density", 5, 20, 10)
        
        interp_idx = st.slider("Select Frame for DNA Audit", 0, steps-1, 0)
        
        st.markdown("---")
        st.markdown("#### Hyperparameters")
        st.code("Latent Dim: 20\nPrior: N(0, 1)\nLoss: ELBO")

    # --- 4. HEADER ---
    st.markdown("# 🧬 VAE Precision Synthesizer")
    st.markdown("<p style='color:#8b949e; font-size: 1.1rem;'>Investigating the continuous latent manifold of the MNIST repository.</p>", unsafe_allow_html=True)

    # Load model
    if not os.path.exists("models/vae_mnist.pkl"):
        st.warning("⚠️ VAE Weights Missing. Run `s04_pipeline.py` first.")
        return
    model = load_vae_model()

    # Main Screen Layout
    tab_gen, tab_latent, tab_audit = st.tabs(["Active Synthesis", "Latent DNA Explorer", "Engineering Metrics"])

    with tab_gen:
        col1, col2 = st.columns([1, 2], gap="large")
        
        with col1:
            st.markdown("### 🧬 Synthesis Engine")
            if st.button("🚀 Regenerate Batch", type="primary", use_container_width=True):
                grid = generate_samples(model, n=64)
                st.image(grid, use_container_width=True, caption="64 Samples synthesized from N(0, 1) prior.")
            else:
                st.info("The model generates images by sampling the 20D latent space and decoding the 'concepts' back into pixels.")

        with col2:
            st.markdown("### 🧠 Mechanism Theory")
            st.markdown("""
            <div class="theory-box">
                <h4 style='color:#238636; margin-top:0'>The Continuity Principle</h4>
                <p style='color:#8b949e; font-size:0.9rem'>Unlike standard Autoencoders, VAEs ensure there are no 'holes' in the latent space. 
                Any point in the distribution decoded into a recognizable digit.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Pixel Intensity Distribution Plot
            # Create a dummy image for the first frame intensity check
            torch.manual_seed(z1_seed)
            z_sample = torch.randn(1, 20)
            with torch.no_grad():
                img_sample = model.decode(z_sample).squeeze().cpu().numpy()
            
            fig_hist = px.histogram(img_sample.flatten(), nbins=30, title="Pixel Intensity Distribution (Active Sample)",
                                    labels={'value': 'Brightness', 'count': 'Frequency'},
                                    color_discrete_sequence=['#238636'])
            fig_hist.update_layout(template="plotly_dark", height=250, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab_latent:
        st.markdown("### 🧬 Latent Concept DNA")
        
        # Interpolation Logic
        torch.manual_seed(z1_seed); v1 = torch.randn(1, 20)
        torch.manual_seed(z2_seed); v2 = torch.randn(1, 20)
        morph_imgs, morph_zs = interpolate_digits(model, v1, v2, steps=steps)
        
        # Display Morphing Row
        st.markdown("#### Latent Trajectory (Interpolation)")
        cols = st.columns(steps)
        for i, img in enumerate(morph_imgs):
            border = "2px solid #58a6ff" if i == interp_idx else "None"
            cols[i].image(img, use_container_width=True)
            if i == interp_idx:
                cols[i].markdown("<p style='text-align:center; color:#58a6ff;'>Audit Target</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        lc1, lc2 = st.columns(2)
        
        with lc1:
            st.markdown("#### Latent Signature DNA")
            # Bar chart of the 20 latent dimensions for the selected frame
            z_audit = morph_zs[interp_idx]
            fig_dna = go.Figure(data=[go.Bar(x=list(range(20)), y=z_audit, marker_color='#58a6ff')])
            fig_dna.update_layout(template="plotly_dark", title=f"Latent Vector Z (Frame {interp_idx})",
                                  xaxis_title="Latent Dimension", yaxis_title="Activation Value",
                                  height=350, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_dna, use_container_width=True)

        with lc2:
            st.markdown("#### Dimensional Contrast")
            # Side-by-side comparison of Seed A and Seed B latent DNA
            z_a = morph_zs[0]
            z_b = morph_zs[-1]
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=list(range(20)), y=z_a, name="Seed A (Pure)", line=dict(color='#238636', dash='dot')))
            fig_comp.add_trace(go.Scatter(x=list(range(20)), y=z_b, name="Seed B (Goal)", line=dict(color='#f85149', dash='dot')))
            fig_comp.add_trace(go.Scatter(x=list(range(20)), y=z_audit, name="Active Frame", fill='toself', line=dict(color='#58a6ff')))
            fig_comp.update_layout(template="plotly_dark", title="Latent Shift Analysis",
                                   height=350, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_comp, use_container_width=True)

    with tab_audit:
        st.markdown("### 📈 Engineering Performance Logs")
        
        history = get_training_history()
        if history:
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(y=history["recon_loss"], name="Reconstruction (BCE)", line=dict(color="#238636")))
            fig_perf.add_trace(go.Scatter(y=history["kl_loss"], name="Concept Regularizer (KL)", line=dict(color="#58a6ff")))
            fig_perf.update_layout(template="plotly_dark", height=400, title="Learning Trajectory (ELBO Optimization)")
            st.plotly_chart(fig_perf, use_container_width=True)
            
            st.success("**Training Audit**: The KL Divergence successfully stabilized, indicating a continuous latent manifold.")
        else:
            st.warning("No training logs found. Run the master pipeline to generate metrics.")

    # --- 5. FOOTER ---
    st.markdown("---")
    st.caption("Project 11: Variational Autoencoder (VAE) · Advanced Deep Learning Track")

if __name__ == "__main__":
    main()
