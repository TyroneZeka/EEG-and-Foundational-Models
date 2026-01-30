import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path
import logging
import mne

# --- Setup Paths and Logging ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Main Visualization Class ---

class BIOT_DirectVisualizer:
    """Visualize BIOT's learned filters directly from a checkpoint."""
    
    def __init__(self, checkpoint_path):
        # The BIOT .pth file is the state_dict itself, not a checkpoint object
        self.state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.sampling_rate = 250 # Assuming this based on our preprocessing

    def get_spatial_importance(self, n_channels):
        """Extracts the norm of the learned channel embedding for each channel."""
        key = 'biot.channel_tokens.weight'
        if key not in self.state_dict:
            logger.error(f"'{key}' not found in checkpoint. Cannot get spatial importance.")
            return None
        
        # Shape: (n_channels, embedding_dim)
        channel_embeddings = self.state_dict[key].numpy()
        # Take the L2 norm to get a single importance value per channel
        spatial_importance = np.linalg.norm(channel_embeddings, axis=1)
        
        return spatial_importance[:n_channels]

    def get_spectral_importance(self):
        """Extracts the norm of the frequency projection weights."""
        key = 'biot.patch_embedding.projection.weight'
        if key not in self.state_dict:
            logger.error(f"'{key}' not found in checkpoint. Cannot get spectral importance.")
            return None, None
            
        # Shape: (embedding_dim, n_freq_bins)
        freq_weights = self.state_dict[key].numpy()
        # Take the norm across the embedding dimension to get importance per frequency bin
        spectral_importance = np.linalg.norm(freq_weights, axis=0)
        
        # We need to know the n_fft used during training to create the frequency axis
        # Assuming n_fft=200 based on our previous scripts
        n_fft = 200
        freq_axis = np.fft.rfftfreq(n_fft, 1 / self.sampling_rate)
        
        # The projection layer has n_freq = n_fft // 2 + 1 inputs
        # We need to make sure our axis matches the weights dimension
        if len(freq_axis) != len(spectral_importance):
             logger.warning("Mismatch between calculated freq axis and weight dimension. Slicing.")
             freq_axis = freq_axis[:len(spectral_importance)]

        return freq_axis, spectral_importance

    def plot_topomap(self, channel_values, ch_names, ax, title):
        """Uses MNE to plot a high-quality topographic map."""
        if channel_values is None or len(channel_values) == 0:
            ax.text(0.5, 0.5, 'No Spatial Data', ha='center', va='center')
            ax.axis('off')
            return
            
        info = mne.create_info(ch_names=ch_names, sfreq=self.sampling_rate, ch_types='eeg')
        info.set_montage('standard_1020', on_missing='warn')

        im, _ = mne.viz.plot_topomap(channel_values, info, axes=ax, show=False, cmap='viridis')
        ax.set_title(title, fontsize=11, fontweight='bold')
        return im

    def plot_spectral(self, freqs, spectral_importance, ax, title):
        """Plots the real spectral importance extracted from the model."""
        if freqs is None or spectral_importance is None:
            ax.text(0.5, 0.5, 'No Spectral Data', ha='center', va='center')
            ax.axis('off')
            return

        ax.plot(freqs, spectral_importance, 'r-', linewidth=2)
        ax.fill_between(freqs, spectral_importance, alpha=0.3, color='darkred')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Weight Norm', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 40)
        ax.axvspan(8, 12, alpha=0.1, color='blue', label='Alpha')
        ax.axvspan(13, 30, alpha=0.1, color='green', label='Beta')
        ax.legend(fontsize=8)

    def generate_figure(self, ch_names, output_path):
        """Generates the combined Figure 3 style plot for the BIOT model."""
        
        spatial_importance = self.get_spatial_importance(len(ch_names))
        freqs, spectral_importance = self.get_spectral_importance()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot Spectral
        self.plot_spectral(freqs, spectral_importance, ax1, 'BIOT - Learned Temporal Importance')
        
        # Plot Spatial
        im_topo = self.plot_topomap(spatial_importance, ch_names, ax2, 'BIOT - Learned Spatial Importance')
        
        if im_topo:
            fig.colorbar(im_topo, ax=ax2, shrink=0.8, label='Channel Embedding Norm')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"✓ Saved BIOT Figure 3 visualization to {output_path}")


def generate_for_all_datasets():
    """Loops through all trained BIOT models and generates visualizations."""
    
    datasets = {
        'BCI_IV_2a': {
            'ch_names': [
                'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
                'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
            ]
        },
        'BNCI2015_001': {
            'ch_names': ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CPz', 'CP4']
        }
    }
    
    for dataset_name, config in datasets.items():
        logger.info(f"\n{'='*80}\nGenerating BIOT Figure 3 for {dataset_name}\n{'='*80}")
        
        model_path = f'models/biot/{dataset_name}_fine_tune.pth'
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}. Skipping.")
            continue
            
        try:
            visualizer = BIOT_DirectVisualizer(model_path)
            output_path = f"analysis/biot_filters/{dataset_name}/figure3_style_plot.png"
            visualizer.generate_figure(config['ch_names'], output_path)
        except Exception as e:
            logger.error(f"Failed to generate plot for {dataset_name}. Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    generate_for_all_datasets()
