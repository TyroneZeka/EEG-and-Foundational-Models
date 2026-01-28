#!/usr/bin/env python
"""
Figure 3 Reproduction: Model Interpretability Analysis
Generates:
1. t-SNE visualizations of learned representations
2. Topographic brain maps (spatial filters)
3. Spectral plots (temporal/frequency response)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from scipy.fft import fft
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import logging
from tqdm import tqdm
import mne
from mne.viz import plot_topomap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.eegnet import EEGNet
from data.load_data import DataLoader as EEGDataLoader
from data.preprocessing import EEGPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterpretabilityAnalyzer:
    """Analyze learned patterns and generate Figure 3 visualizations."""
    
    def __init__(self, model_path, dataset_name, n_channels, n_classes, sampling_rate=250):
        """
        Args:
            model_path: Path to saved model weights
            dataset_name: Name of dataset (BCI_IV_2a, BCI_IV_2b, PhysioNet_MI)
            n_channels: Number of EEG channels
            n_classes: Number of classes
            sampling_rate: EEG sampling rate in Hz
        """
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.sampling_rate = sampling_rate
        
        # Standard 10-20 electrode positions for 22 channels
        self.channel_names_22 = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C3', 'C1', 'Cz', 'C2', 'C4',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'P1', 'Pz', 'P2',
            'POz',
            'O1', 'O2'
        ]
        
        # 64-channel electrode positions (PhysioNet)
        self.channel_names_64 = self._create_64_channel_names()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        
    def _create_64_channel_names(self):
        """Create standard 64-channel electrode names."""
        # Standard 10-5 system subset
        return [f'Ch{i}' for i in range(1, 65)]
    
    def load_model(self, n_samples):
        """Load trained EEGNet model."""
        self.model = EEGNet(
            self.n_channels, 
            self.n_classes, 
            n_samples,
            F1=16, 
            F2=32, 
            D=4
        ).to(self.device)
        
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            logger.info(f"Loaded model from {self.model_path}")
        else:
            logger.warning(f"Model file not found: {self.model_path}")
            return False
        
        self.model.eval()
        return True
    
    def extract_intermediate_features(self, X, layer='block2_out'):
        """
        Extract intermediate features from model for t-SNE.
        
        Args:
            X: Input array (n_samples, n_channels, n_timepoints)
            layer: Which layer to extract from ('block1_out', 'block2_out', 'block3_out')
        
        Returns:
            features: Extracted features (n_samples, n_features)
        """
        features_list = []
        
        with torch.no_grad():
            for i in range(0, len(X), 32):
                batch = X[i:i+32]
                x = torch.from_numpy(batch).float().to(self.device)
                
                # Add channel dimension
                x = x.unsqueeze(1)
                
                # Forward pass with intermediate extraction
                x = self.model.conv1(x)
                x = self.model.batchnorm1(x)
                
                if layer == 'block1_out':
                    features = x.cpu().numpy()
                    features = features.reshape(features.shape[0], -1)
                    features_list.append(features)
                    continue
                
                x = self.model.depthwise(x)
                x = self.model.batchnorm2(x)
                x = self.model.activation(x)
                x = self.model.pool1(x)
                x = self.model.dropout1(x)
                
                if layer == 'block2_out':
                    features = x.cpu().numpy()
                    features = features.reshape(features.shape[0], -1)
                    features_list.append(features)
                    continue
                
                x = self.model.separable_depth(x)
                x = self.model.separable_point(x)
                x = self.model.batchnorm3(x)
                x = self.model.activation(x)
                x = self.model.pool2(x)
                
                features = x.cpu().numpy()
                features = features.reshape(features.shape[0], -1)
                features_list.append(features)
        
        return np.vstack(features_list)
    
    def generate_tsne_visualization(self, X, y, output_dir, perplexity=30):
        """Generate t-SNE visualization of learned representations."""
        logger.info(f"Extracting intermediate features from {self.dataset_name}...")
        features = self.extract_intermediate_features(X, layer='block3_out')
        
        # Normalize features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        logger.info("Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, len(X)//3))
        features_2d = tsne.fit_transform(features)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Color map for classes
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_classes))
        class_names = self._get_class_names()
        
        for class_idx in range(self.n_classes):
            mask = y == class_idx
            plt.scatter(
                features_2d[mask, 0], 
                features_2d[mask, 1],
                c=[colors[class_idx]],
                label=class_names[class_idx],
                alpha=0.7,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
        
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.title(f't-SNE Visualization - {self.dataset_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'tsne_{self.dataset_name}.png')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved t-SNE visualization to {output_path}")
        plt.close()
        
        return features_2d
    
    def extract_spatial_filters(self):
        """
        Extract spatial filters from the first convolution layer.
        These represent what the model "sees" spatially.
        """
        # Get temporal conv weights: (F1, 1, 1, kernel_length)
        temporal_weights = self.model.conv1.weight.data.cpu().numpy()
        
        # Get depthwise weights: (F1*D, 1, n_channels, 1)
        spatial_weights = self.model.depthwise.weight.data.cpu().numpy()
        
        return temporal_weights, spatial_weights
    
    def compute_spectral_response(self, kernel_length=64):
        """
        Compute frequency response of temporal filters.
        Shows which frequencies the model attends to.
        """
        temporal_weights, _ = self.extract_spatial_filters()
        
        # Average across filters
        avg_kernel = np.mean(temporal_weights, axis=0).squeeze()  # (kernel_length,)
        
        # Compute FFT
        freqs = np.fft.fftfreq(kernel_length, 1/self.sampling_rate)[:kernel_length//2]
        fft_response = np.abs(np.fft.fft(avg_kernel)[:kernel_length//2])
        
        return freqs, fft_response
    
    # Replace your existing function with this one
    def generate_topographic_maps(self, output_dir):
        """
        Generates topographic brain maps for individual spatial filters.
        This shows the specific spatial patterns learned by the model.
        """
        if self.n_channels not in [22, 64]:
            logger.warning("Unsupported channel count for topomaps. Skipping.")
            return

        # Use the appropriate channel names and create MNE info
        channel_names = self.channel_names_22 if self.n_channels == 22 else self.channel_names_64
        info = mne.create_info(ch_names=channel_names, sfreq=self.sampling_rate, ch_types='eeg')
        # This is crucial for plotting on a head shape
        info.set_montage('standard_1020', on_missing='ignore')

        logger.info("Generating topographic maps for spatial filters...")

        # --- THIS IS THE KEY CHANGE ---
        # Extract the spatial filters from the depthwise convolution layer
        # Shape: (F1 * D, 1, n_channels, 1)
        _, spatial_weights = self.extract_spatial_filters()
        # Let's visualize the first F1 filters. Squeeze to shape (F1*D, n_channels)
        spatial_filters = spatial_weights.squeeze()

        # Determine how many filters to plot (e.g., the first F1=16 filters)
        n_filters_to_plot = self.model.F1 

        # Setup the plot grid
        n_cols = 4
        n_rows = int(np.ceil(n_filters_to_plot / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

        # Find the global min/max across all filters for consistent color scaling
        v_max = np.max(np.abs(spatial_filters[:n_filters_to_plot]))
        v_min = -v_max

        for i, ax in enumerate(axes.flat):
            if i < n_filters_to_plot:
                # Get the pattern for the current filter
                filter_pattern = spatial_filters[i]

                # Plot the topomap for THIS filter
                im, _ = plot_topomap(
                    filter_pattern,
                    info,
                    axes=ax,
                    show=False,
                    cmap='RdBu_r', # Red-Blue colormap for intensity
                    vmin=v_min,    # Use symmetric min/max for a balanced colormap
                    vmax=v_max
                )
                ax.set_title(f'Filter {i+1}', fontsize=10)
            else:
                # Hide unused subplots
                ax.axis('off')

        # Add a single colorbar for the whole figure
        cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Filter Weight Intensity')

        plt.suptitle(f'Spatial Filters (First {n_filters_to_plot} of {len(spatial_filters)}) - {self.dataset_name}', fontsize=16, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 0.9, 0.95]) # Adjust layout to make space for colorbar and title

        output_path = os.path.join(output_dir, f'topomaps_{self.dataset_name}.png')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved topographic maps to {output_path}")
        plt.close()


    
    # Replace your existing function with this one
    def generate_spectral_plots(self, output_dir):
        """
        Generates spectral response plots for individual temporal filters.
        """
        logger.info("Generating spectral plots for temporal filters...")
        
        # Shape: (F1, 1, 1, kernel_length)
        temporal_weights, _ = self.extract_spatial_filters()
        # Squeeze to (F1, kernel_length)
        temporal_kernels = temporal_weights.squeeze()
        
        kernel_length = temporal_kernels.shape[1]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        
        # --- THIS IS THE KEY CHANGE ---
        # Compute and plot FFT for each individual kernel
        for kernel in temporal_kernels:
            freqs = np.fft.rfftfreq(kernel_length, 1/self.sampling_rate)
            fft_response = np.abs(np.fft.rfft(kernel))
            
            # Plot each filter's response with low opacity to see the density
            ax.plot(freqs, fft_response, linewidth=1.5, alpha=0.6)
            
        # --- The rest is for styling the plot ---
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Filter Response (Magnitude)', fontsize=12)
        ax.set_title(f'Spectral Response of all {len(temporal_kernels)} Temporal Filters - {self.dataset_name}',
                      fontsize=14, fontweight='bold')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlim([0, 40]) # Focus on the most relevant EEG frequency range
        ax.set_ylim(bottom=0)
        
        # Add labels for common EEG bands for context
        bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (13, 30)}
        for band, (low, high) in bands.items():
            ax.axvspan(low, high, alpha=0.1, color=plt.cm.viridis((high-low)/30), label=f'{band} ({low}-{high} Hz)')
        
        ax.legend(loc='best', fontsize=10)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'spectral_{self.dataset_name}.png')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved spectral plot to {output_path}")
        plt.close()
    
    
    def _get_class_names(self):
        """Get class names for the dataset."""
        class_names_dict = {
            'BCI_IV_2a': ['Left Hand', 'Right Hand', 'Feet', 'Tongue'],
            'BCI_IV_2b': ['Left Hand', 'Right Hand'],
            'PhysioNet_MI': ['Feet', 'Hands', 'Rest']
        }
        return class_names_dict.get(self.dataset_name, [f'Class {i}' for i in range(self.n_classes)])
    
    def analyze_all(self, X, y, output_dir):
        """Generate all visualizations."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Analyzing {self.dataset_name}")
        logger.info(f"{'='*80}")
        
        # t-SNE
        self.generate_tsne_visualization(X, y, output_dir)
        
        # Topographic maps
        self.generate_topographic_maps(output_dir)
        
        # Spectral plots
        self.generate_spectral_plots(output_dir)
        
        logger.info(f"Analysis complete for {self.dataset_name}\n")


def run_figure3_analysis():
    """Run complete Figure 3 reproduction analysis on all datasets."""
    
    # Configuration
    datasets_config = [
        {
            'name': 'BCI_IV_2a',
            'n_channels': 22,
            'n_classes': 4,
            'loader_func': 'load_bci_2a'
        },
        {
            'name': 'BCI_IV_2b',
            'n_channels': 22,
            'n_classes': 2,
            'loader_func': 'load_bci_2b'
        },
        {
            'name': 'PhysioNet_MI',
            'n_channels': 64,
            'n_classes': 3,
            'loader_func': 'load_physionet_mi'
        }
    ]
    
    output_base_dir = 'analysis/figure3_results'
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading datasets...")
    data_loader = EEGDataLoader()
    preprocessor = EEGPreprocessor(sampling_rate=250)
    
    for config in datasets_config:
        dataset_name = config['name']
        n_channels = config['n_channels']
        n_classes = config['n_classes']
        
        logger.info(f"\nProcessing {dataset_name}...")
        
        # Load dataset
        try:
            load_func = getattr(data_loader, config['loader_func'])
            data = load_func()
            X, y = data['X'], data['y']
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            continue
        
        # Preprocess
        X, y = preprocessor.preprocess(X, y)
        
        # Create analyzer
        n_samples = X.shape[2]
        model_path = f'experiments/task1_eegnet/{dataset_name.lower()}_best_model.pth'
        
        analyzer = InterpretabilityAnalyzer(
            model_path=model_path,
            dataset_name=dataset_name,
            n_channels=n_channels,
            n_classes=n_classes,
            sampling_rate=250
        )
        
        # Load model
        if not analyzer.load_model(n_samples):
            logger.warning(f"Skipping {dataset_name} - model not found")
            continue
        
        # Run analysis
        dataset_output_dir = os.path.join(output_base_dir, dataset_name)
        analyzer.analyze_all(X, y, dataset_output_dir)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Figure 3 Analysis Complete!")
    logger.info(f"Results saved to: {output_base_dir}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    run_figure3_analysis()
