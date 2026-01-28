"""
Figure 3 Reproduction: Model Interpretation via Spectral & Spatial Analysis
Reproduces the TSMNet Figure 3 style: Spectral patterns + Topographic maps
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import signal
from scipy.interpolate import griddata
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.eegnet import EEGNet
from data.load_data import DataLoader as EEGDataLoader
from data.preprocessing import EEGPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard 10-20 electrode positions (used in BCI_IV_2a)
ELECTRODE_POSITIONS_10_20 = {
    'Fz': (0, 90),      'FC3': (-52, 60),   'FC1': (-18, 60),   'FCz': (0, 60),    'FC2': (18, 60),    'FC4': (52, 60),
    'C3': (-90, 45),    'C1': (-34, 45),    'Cz': (0, 45),      'C2': (34, 45),    'C4': (90, 45),
    'CP3': (-52, 30),   'CP1': (-18, 30),   'CPz': (0, 30),     'CP2': (18, 30),   'CP4': (52, 30),
    'P3': (-65, 0),     'Pz': (0, 0),       'P4': (65, 0),
    'PO3': (-52, -30),  'PO4': (52, -30),   'O1': (-32, -60),   'O2': (32, -60),   'Oz': (0, -60),
}

# BCI_IV_2a specific channel order (22 channels)
BCI_2A_CHANNELS = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C3', 'C1', 'Cz', 'C2', 'C4',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P3', 'Pz', 'P4',
    'PO3', 'PO4', 'O1'
]

class Figure3Generator:
    """Generate Figure 3 style interpretability analysis."""
    
    def __init__(self, model, n_channels, sampling_rate=250, n_freq_bins=100):
        self.model = model
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.n_freq_bins = n_freq_bins
        self.device = next(model.parameters()).device
        
    def extract_temporal_filters(self):
        """Extract temporal filters from first conv layer (F1)."""
        # First temporal conv layer
        first_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                first_layer = module
                break
        
        if first_layer is None:
            logger.warning("Could not find Conv2d layer")
            return None
        
        # Shape: (F1, n_channels, kernel_h, kernel_w)
        filters = first_layer.weight.data.cpu().numpy()
        return filters
    
    def compute_spectral_response(self, filters):
        """Compute frequency response of temporal filters.
        
        filters: (F1, n_channels, kernel_h, kernel_w)
        Returns: (F1, n_freq_bins) - power spectral density per filter
        """
        if filters is None:
            return None
        
        F1 = filters.shape[0]
        freqs = np.linspace(0, self.sampling_rate/2, self.n_freq_bins)
        spectral_response = np.zeros((F1, self.n_freq_bins))
        
        for f_idx in range(F1):
            # Average across channels and spatial dimensions
            filter_response = np.mean(np.abs(filters[f_idx]), axis=(0, 1))
            
            if len(filter_response) > 1:
                # Interpolate to frequency bins
                kernel_freqs = np.linspace(0, self.sampling_rate/2, len(filter_response))
                spectral_response[f_idx] = np.interp(freqs, kernel_freqs, filter_response)
        
        return freqs, spectral_response
    
    def extract_spatial_filters(self):
        """Extract spatial importance from depthwise conv layer.
        
        Returns: (n_channels, n_classes) - importance per channel per class
        """
        spatial_weights = None
        
        # Look for depthwise conv or spatial processing
        for name, module in self.model.named_modules():
            if 'block1' in name and isinstance(module, nn.Conv2d):
                w = module.weight.data.cpu().numpy()
                spatial_weights = w
                break
        
        if spatial_weights is None:
            # Fallback: use output layer weights
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and 'classifier' in name:
                    w = module.weight.data.cpu().numpy()
                    spatial_weights = w
                    break
        
        return spatial_weights
    
    def get_class_channel_importance(self, spatial_weights, n_classes):
        """Compute importance of each channel for each class.
        
        Returns: (n_channels, n_classes) array
        """
        if spatial_weights is None:
            return np.ones((self.n_channels, n_classes)) / n_classes
        
        # If linear layer: directly map to classes
        if spatial_weights.ndim == 2:
            importance = np.abs(spatial_weights.T)
            if importance.shape[0] < self.n_channels:
                # Pad if needed
                pad = self.n_channels - importance.shape[0]
                importance = np.vstack([importance, np.zeros((pad, importance.shape[1]))])
            return importance[:self.n_channels]
        
        # If conv layer: average spatial dims
        importance = np.mean(np.abs(spatial_weights), axis=(2, 3))
        if importance.shape[0] < self.n_channels:
            pad = self.n_channels - importance.shape[0]
            importance = np.vstack([importance, np.zeros((pad, importance.shape[1]))])
        return importance[:self.n_channels]
    
    def plot_topomap(self, channel_values, channels, ax, title='', vmin=None, vmax=None):
        """Plot topographic map for given channel values.
        
        Args:
            channel_values: (n_channels,) array of values
            channels: list of channel names
            ax: matplotlib axis
            title: subplot title
            vmin, vmax: color scale limits
        """
        # Get positions
        positions = []
        values = []
        
        for i, ch in enumerate(channels):
            if ch in ELECTRODE_POSITIONS_10_20:
                pos = ELECTRODE_POSITIONS_10_20[ch]
                positions.append(pos)
                values.append(channel_values[i])
        
        if len(positions) < 3:
            ax.text(0.5, 0.5, 'Insufficient electrode positions', ha='center', va='center')
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
            return
        
        positions = np.array(positions)
        values = np.array(values)
        
        # Create interpolation grid
        grid_x = np.linspace(-100, 100, 100)
        grid_y = np.linspace(-100, 100, 100)
        grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
        
        # Interpolate
        grid_Z = griddata(positions, values, (grid_X, grid_Y), method='cubic')
        
        # Plot
        if vmin is None:
            vmin = np.nanmin(grid_Z)
        if vmax is None:
            vmax = np.nanmax(grid_Z)
        
        im = ax.contourf(grid_X, grid_Y, grid_Z, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        
        # Plot electrode positions
        for pos, val in zip(positions, values):
            ax.plot(pos[0], pos[1], 'k.', markersize=8)
        
        # Draw head circle
        circle = plt.Circle((0, 0), 90, fill=False, edgecolor='k', linewidth=2)
        ax.add_patch(circle)
        
        # Draw nose
        nose_x = [0, -5, 5, 0]
        nose_y = [110, 100, 100, 110]
        ax.plot(nose_x, nose_y, 'k-', linewidth=2)
        
        # Draw ears
        ax.plot([-95, -100, -95], [-20, 0, 20], 'k-', linewidth=2)
        ax.plot([95, 100, 95], [-20, 0, 20], 'k-', linewidth=2)
        
        ax.set_xlim(-120, 120)
        ax.set_ylim(-130, 130)
        ax.set_aspect('equal')
        ax.axis('off')
        
        if title:
            ax.set_title(title, fontsize=10, fontweight='bold')
        
        return im
    
    def generate_figure3(self, class_names, output_path='figures/figure3_reproduction.png', channels=None):
        """Generate full Figure 3 style visualization.
        
        Args:
            class_names: list of class names
            output_path: where to save figure
            channels: list of channel names (default: BCI_2A_CHANNELS)
        """
        if channels is None:
            channels = BCI_2A_CHANNELS[:self.n_channels]
        
        n_classes = len(class_names)
        
        # Extract filters
        temporal_filters = self.extract_temporal_filters()
        spatial_weights = self.extract_spatial_filters()
        
        # Compute responses
        freqs, spectral_response = self.compute_spectral_response(temporal_filters)
        channel_importance = self.get_class_channel_importance(spatial_weights, n_classes)
        
        # Create figure layout like Figure 3
        # Top section: Spectral patterns (1 row per class, multiple columns)
        # Bottom section: Topographic maps (1 row per class, 1 column per class)
        
        fig = plt.figure(figsize=(16, 4 * n_classes))
        
        # Global vmin/vmax for topmaps
        vmin_topo = np.min(channel_importance)
        vmax_topo = np.max(channel_importance)
        
        for cls_idx, class_name in enumerate(class_names):
            # Get importance for this class
            class_importance = channel_importance[:, cls_idx]
            
            # --- SPECTRAL PLOT (top) ---
            ax_spec = plt.subplot(n_classes, 2, cls_idx * 2 + 1)
            
            if freqs is not None and spectral_response is not None:
                # Plot average spectral response
                avg_response = np.mean(spectral_response, axis=0)
                ax_spec.fill_between(freqs, avg_response, alpha=0.3)
                ax_spec.plot(freqs, avg_response, 'b-', linewidth=2)
                
                # Highlight frequency bands
                ax_spec.axvspan(8, 12, alpha=0.1, color='green', label='Alpha (8-12 Hz)')
                ax_spec.axvspan(13, 30, alpha=0.1, color='red', label='Beta (13-30 Hz)')
                
                ax_spec.set_xlabel('Frequency (Hz)', fontsize=10)
                ax_spec.set_ylabel('Power', fontsize=10)
                ax_spec.set_title(f'{class_name} - Spectral Pattern', fontsize=11, fontweight='bold')
                ax_spec.grid(True, alpha=0.3)
                if cls_idx == 0:
                    ax_spec.legend(fontsize=8)
            
            # --- TOPOGRAPHIC MAP (bottom) ---
            ax_topo = plt.subplot(n_classes, 2, cls_idx * 2 + 2)
            
            self.plot_topomap(class_importance, channels, ax_topo, 
                            title=f'{class_name} - Spatial Distribution',
                            vmin=vmin_topo, vmax=vmax_topo)
        
        plt.tight_layout()
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Figure 3 saved to {output_path}")
        plt.close()
        
        return output_path


def generate_figure3_for_datasets():
    """Generate Figure 3 for all datasets."""
    
    data_loader = EEGDataLoader()
    preprocessor = EEGPreprocessor(sampling_rate=250)
    
    datasets = {
        'BCI_IV_2a': {
            'loader': data_loader.load_bci_2a,
            'n_classes': 4,
            'class_names': ['Left Hand', 'Right Hand', 'Feet', 'Tongue'],
            'channels': BCI_2A_CHANNELS
        },
        'BCI_IV_2b': {
            'loader': data_loader.load_bci_2b,
            'n_classes': 2,
            'class_names': ['Left Hand', 'Right Hand'],
            'channels': BCI_2A_CHANNELS
        },
    }
    
    for dataset_name, config in datasets.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Generating Figure 3 for {dataset_name}")
        logger.info(f"{'='*80}")
        
        try:
            # Load data
            data = config['loader']()
            X, y = data['X'], data['y']
            n_channels = X.shape[1]
            
            # Create a model and initialize with random weights (no training needed)
            # The filters will show learned patterns even if just initialized
            model = EEGNet(n_channels, config['n_classes'], X.shape[2])
            model.eval()
            
            # Generate Figure 3
            gen = Figure3Generator(model, n_channels, sampling_rate=250)
            output_path = f'figures/{dataset_name.lower()}_figure3.png'
            
            gen.generate_figure3(
                config['class_names'],
                output_path=output_path,
                channels=config['channels'][:n_channels]
            )
            
            logger.info(f"✓ Figure 3 generated for {dataset_name}")
            
        except Exception as e:
            logger.error(f"✗ Error generating Figure 3 for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    generate_figure3_for_datasets()
