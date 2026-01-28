#!/usr/bin/env python
"""
Figure 3 Reproduction: Model Interpretability Analysis
This version runs analysis on trained models during/after LOSO CV
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
    
    def __init__(self, model, dataset_name, n_channels, n_classes, sampling_rate=250):
        """
        Args:
            model: Trained PyTorch EEGNet model
            dataset_name: Name of dataset (BCI_IV_2a, BCI_IV_2b, PhysioNet_MI)
            n_channels: Number of EEG channels
            n_classes: Number of classes
            sampling_rate: EEG sampling rate in Hz
        """
        self.model = model
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
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.eval()
    
    def extract_intermediate_features(self, X, layer='block3_out'):
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
    
    def generate_tsne_visualization(self, X, y, output_dir, fold_idx=None, perplexity=30):
        """Generate t-SNE visualization of learned representations."""
        logger.info(f"Extracting intermediate features...")
        features = self.extract_intermediate_features(X, layer='block3_out')
        
        # Normalize features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        logger.info("Computing t-SNE (this may take a few minutes)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, max(5, len(X)//3)))
        features_2d = tsne.fit_transform(features)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color map for classes
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_classes))
        class_names = self._get_class_names()
        
        for class_idx in range(self.n_classes):
            mask = y == class_idx
            if np.any(mask):
                ax.scatter(
                    features_2d[mask, 0], 
                    features_2d[mask, 1],
                    c=[colors[class_idx]],
                    label=class_names[class_idx],
                    alpha=0.7,
                    s=100,
                    edgecolors='black',
                    linewidth=0.5
                )
        
        ax.set_xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
        
        fold_str = f' - Fold {fold_idx}' if fold_idx is not None else ''
        ax.set_title(f't-SNE Visualization - {self.dataset_name}{fold_str}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        filename = f'tsne_{self.dataset_name}' + (f'_fold{fold_idx}' if fold_idx else '') + '.png'
        output_path = os.path.join(output_dir, filename)
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
        
        # Pad to power of 2 for better FFT
        n_fft = 2 ** int(np.ceil(np.log2(len(avg_kernel))))
        
        # Compute FFT
        freqs = np.fft.fftfreq(n_fft, 1/self.sampling_rate)[:n_fft//2]
        fft_response = np.abs(np.fft.fft(avg_kernel, n_fft)[:n_fft//2])
        
        # Smooth
        fft_response = signal.savgol_filter(fft_response, window_length=min(11, len(fft_response)//2*2+1), polyorder=3)
        
        return freqs, fft_response
    
    def generate_topographic_maps(self, output_dir, fold_idx=None):
        """
        Generate topographic brain maps showing spatial filter importance.
        For each class, shows which brain regions are most important.
        """
        if self.n_channels == 22:
            channel_names = self.channel_names_22
        else:
            # For 64 channels, use standard names
            channel_names = [f'Ch{i}' for i in range(1, self.n_channels + 1)]
        
        # Create MNE info structure
        try:
            info = mne.create_info(ch_names=channel_names[:self.n_channels], sfreq=self.sampling_rate, ch_types='eeg')
        except Exception as e:
            logger.warning(f"Could not create MNE info: {e}")
            return
        
        logger.info("Generating topographic maps...")
        
        # Use first layer weights as spatial importance
        temporal_weights, spatial_weights = self.extract_spatial_filters()
        
        # Average temporal filters per channel
        spatial_importance = np.mean(np.abs(spatial_weights), axis=(0, 2, 3))[:self.n_channels]
        
        # Normalize
        if spatial_importance.max() > 0:
            spatial_importance = (spatial_importance - spatial_importance.min()) / (spatial_importance.max() - spatial_importance.min())
        
        fig = plt.figure(figsize=(5*min(4, self.n_classes), 5))
        
        class_names = self._get_class_names()
        
        for class_idx in range(min(4, self.n_classes)):
            ax = plt.subplot(1, min(4, self.n_classes), class_idx + 1)
            
            try:
                im, _ = plot_topomap(
                    spatial_importance, 
                    info,
                    axes=ax,
                    show=False,
                    cmap='RdBu_r',
                    vmin=-np.max(np.abs(spatial_importance)),
                    vmax=np.max(np.abs(spatial_importance))
                )
                ax.set_title(f'{class_names[class_idx]}', fontsize=11, fontweight='bold')
            except Exception as e:
                logger.debug(f"Could not generate topomap: {e}")
                ax.text(0.5, 0.5, 'Topomap\nUnavailable', ha='center', va='center', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        
        fold_str = f' - Fold {fold_idx}' if fold_idx is not None else ''
        plt.suptitle(f'Spatial Filters - {self.dataset_name}{fold_str}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filename = f'topomaps_{self.dataset_name}' + (f'_fold{fold_idx}' if fold_idx else '') + '.png'
        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved topographic maps to {output_path}")
        plt.close()
    
    def generate_spectral_plots(self, output_dir, fold_idx=None):
        """Generate spectral (frequency) response plots of temporal filters."""
        logger.info("Generating spectral plots...")
        
        freqs, fft_response = self.compute_spectral_response()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(freqs, fft_response, linewidth=2.5, color='steelblue', label='Filter Response')
        ax.fill_between(freqs, fft_response, alpha=0.3, color='steelblue')
        
        # Mark important frequency bands
        alpha_band = (8, 12)
        beta_band = (13, 30)
        gamma_band = (30, 50)
        
        ax.axvspan(*alpha_band, alpha=0.1, color='red', label='Alpha (8-12 Hz)')
        ax.axvspan(*beta_band, alpha=0.1, color='green', label='Beta (13-30 Hz)')
        ax.axvspan(*gamma_band, alpha=0.1, color='blue', label='Gamma (30-50 Hz)')
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Filter Response Magnitude', fontsize=12, fontweight='bold')
        
        fold_str = f' - Fold {fold_idx}' if fold_idx is not None else ''
        ax.set_title(f'Spectral Response of Temporal Filters - {self.dataset_name}{fold_str}', 
                      fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, 50])
        
        plt.tight_layout()
        
        filename = f'spectral_{self.dataset_name}' + (f'_fold{fold_idx}' if fold_idx else '') + '.png'
        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    
    def analyze_fold(self, X, y, output_dir, fold_idx):
        """Generate all visualizations for a specific fold."""
        logger.info(f"Analyzing fold {fold_idx}...")
        
        # t-SNE
        self.generate_tsne_visualization(X, y, output_dir, fold_idx=fold_idx)
        
        # Topographic maps
        self.generate_topographic_maps(output_dir, fold_idx=fold_idx)
        
        # Spectral plots
        self.generate_spectral_plots(output_dir, fold_idx=fold_idx)
    
    def analyze_aggregate(self, X, y, output_dir):
        """Generate visualizations for aggregate data (all folds combined)."""
        logger.info(f"Analyzing aggregate data (all samples)...")
        
        # t-SNE
        self.generate_tsne_visualization(X, y, output_dir, fold_idx=None)
        
        # Topographic maps
        self.generate_topographic_maps(output_dir, fold_idx=None)
        
        # Spectral plots
        self.generate_spectral_plots(output_dir, fold_idx=None)


# Export for use in training scripts
__all__ = ['InterpretabilityAnalyzer']
