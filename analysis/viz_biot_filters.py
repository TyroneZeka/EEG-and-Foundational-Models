import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mne
from pathlib import Path
import logging

# --- Setup Paths and Logging ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.biot import BIOTClassifier
from data.load_data import DataLoader as EEGDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Visualization Functions ---

def visualize_biot_spatial_filters(model, ch_names, sfreq, output_path):
    """Visualizes the BIOT model's learned channel embeddings as a topomap."""
    logging.info(f"Visualizing BIOT spatial filters for {len(ch_names)} channels...")
    try:
        channel_embeddings = model.biot.channel_tokens.weight.data.cpu().numpy()
    except AttributeError:
        logging.error("Could not find 'model.biot.channel_tokens'. Check model structure.")
        return

    channel_importance = np.linalg.norm(channel_embeddings, axis=1)

    # Make sure we only use the importance values for the channels we have
    if len(channel_importance) < len(ch_names):
        logging.error("Model has fewer channel embeddings than expected. Cannot create topomap.")
        return
    channel_importance = channel_importance[:len(ch_names)]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    info.set_montage('standard_1020', on_missing='warn')

    fig, ax = plt.subplots(figsize=(7, 6))
    im, _ = mne.viz.plot_topomap(channel_importance, info, axes=ax, show=False, cmap='viridis')
    ax.set_title('BIOT Spatial Importance (Channel Embedding Norm)', fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8, label='Embedding L2 Norm')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"✓ Saved BIOT spatial filter topomap to {output_path}")

def visualize_biot_temporal_filters(model, sfreq, output_path):
    """Visualizes the weights of the PatchFrequencyEmbedding layer."""
    logging.info("Visualizing BIOT temporal filters...")
    try:
        freq_weights = model.biot.patch_embedding.projection.weight.data.cpu().numpy()
        n_fft = model.biot.n_fft
    except AttributeError:
        logging.error("Could not find required attributes for temporal visualization.")
        return

    freq_importance = np.linalg.norm(freq_weights, axis=0)
    freq_bins = np.fft.rfftfreq(n_fft, 1 / sfreq)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(freq_bins, freq_importance, color='darkred', linewidth=2)
    ax.fill_between(freq_bins, freq_importance, color='darkred', alpha=0.3)
    ax.set_title('BIOT Temporal Importance (Frequency Embedding Weights)', fontsize=16)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Weight Norm (Importance)')
    ax.set_xlim(1, 40)
    ax.grid(True, linestyle='--')
    ax.axvspan(8, 12, color='blue', alpha=0.1, label='Alpha Band (8-12 Hz)')
    ax.axvspan(13, 30, color='green', alpha=0.1, label='Beta Band (13-30 Hz)')
    ax.legend()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"✓ Saved BIOT temporal filter plot to {output_path}")

# --- Main Execution Logic ---

def main():
    """
    Main function to load trained BIOT models and visualize their input filters.
    """
    # --- MASTER CONFIGURATION ---
    BASE_MODEL_PATH = 'models/biot'
    OUTPUT_DIR = 'analysis/biot_filters'
    
    # Define the datasets you want to visualize
    DATASETS_TO_VISUALIZE = {
        'BCI_IV_2a': {
            'loader_func': 'load_bci_2a',
            'n_classes': 4,
            'ch_names': [
                'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
                'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
            ]
        },
        'BNCI2015_001': {
            'loader_func': 'load_bnci2015_001',
            'n_classes': 2,
            'ch_names': ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CPz', 'CP4']
        }
    }
    
    MODEL_CONFIG = {
        'emb_size': 256, 'heads': 8, 'depth': 4, 'n_fft': 200, 'hop_length': 20
    }
    # --- END OF CONFIGURATION ---

    data_loader_manager = EEGDataLoader()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for name, config in DATASETS_TO_VISUALIZE.items():
        logging.info(f"\n{'='*80}\nVisualizing for Dataset: {name}\n{'='*80}")
        
        # Define model path
        model_path = Path(BASE_MODEL_PATH) / f"{name}_fine_tune.pth"
        output_subdir = Path(OUTPUT_DIR) / name
        os.makedirs(output_subdir, exist_ok=True)

        if not model_path.exists():
            logging.error(f"Model not found at {model_path}. Skipping.")
            continue
        
        # Load data to get info, but we don't need to preprocess it
        loader = getattr(data_loader_manager, config['loader_func'])
        data = loader()
        sfreq = data['sampling_rate']
        n_channels = data['channels']

        # Initialize model and load state
        current_model_config = {
            **MODEL_CONFIG,
            'n_channels': n_channels,
            'n_classes': config['n_classes']
        }
        
        model = BIOTClassifier(**current_model_config).to(device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except RuntimeError as e:
            logging.error(f"Failed to load model for {name}. Size mismatch? Error: {e}")
            continue

        # Run visualizations
        visualize_biot_spatial_filters(model, config['ch_names'], sfreq, output_path=output_subdir / "spatial_filters.png")
        visualize_biot_temporal_filters(model, sfreq, output_path=output_subdir / "temporal_filters.png")

if __name__ == "__main__":
    main()
