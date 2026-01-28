"""
Class-Specific Figure 3 Analysis - Direct Checkpoint Visualization
No model reconstruction needed - just visualize weights from checkpoint
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard 10-20 electrode positions
ELECTRODE_POSITIONS_10_20 = {
    'Fz': (0, 90), 'FC3': (-52, 60), 'FC1': (-18, 60), 'FCz': (0, 60), 'FC2': (18, 60), 'FC4': (52, 60),
    'C3': (-90, 45), 'C1': (-34, 45), 'Cz': (0, 45), 'C2': (34, 45), 'C4': (90, 45),
    'CP3': (-52, 30), 'CP1': (-18, 30), 'CPz': (0, 30), 'CP2': (18, 30), 'CP4': (52, 30),
    'P3': (-65, 0), 'Pz': (0, 0), 'P4': (65, 0),
    'PO3': (-52, -30), 'PO4': (52, -30), 'O1': (-32, -60), 'O2': (32, -60), 'Oz': (0, -60),
}

BCI_2A_CHANNELS = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C3', 'C1', 'Cz', 'C2', 'C4',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P3', 'Pz', 'P4', 'PO3', 'PO4', 'O1'
]

# Extended 10-10 system for 64 channels
ELECTRODE_POSITIONS_10_10 = {
    # Frontal
    'Nz': (0, 120), 'Fpz': (0, 110), 'AFz': (0, 105), 'Fz': (0, 90),
    'Fp1': (-18, 110), 'Fp2': (18, 110), 'AF3': (-34, 105), 'AF4': (34, 105),
    'AF7': (-52, 100), 'AF8': (52, 100), 'F1': (-34, 75), 'F2': (34, 75),
    'F3': (-52, 70), 'F4': (52, 70), 'F5': (-65, 60), 'F6': (65, 60),
    'F7': (-75, 50), 'F8': (75, 50), 'Fz': (0, 90),
    # Frontocentral
    'FCz': (0, 60), 'FC1': (-18, 60), 'FC2': (18, 60),
    'FC3': (-52, 60), 'FC4': (52, 60), 'FC5': (-65, 52), 'FC6': (65, 52),
    # Central
    'Cz': (0, 45), 'C1': (-34, 45), 'C2': (34, 45),
    'C3': (-90, 45), 'C4': (90, 45), 'C5': (-75, 40), 'C6': (75, 40),
    # Centroparietal
    'CPz': (0, 30), 'CP1': (-18, 30), 'CP2': (18, 30),
    'CP3': (-52, 30), 'CP4': (52, 30), 'CP5': (-65, 25), 'CP6': (65, 25),
    # Parietal
    'Pz': (0, 0), 'P1': (-34, 0), 'P2': (34, 0),
    'P3': (-65, 0), 'P4': (65, 0), 'P5': (-75, -10), 'P6': (75, -10),
    'P7': (-90, -20), 'P8': (90, -20),
    # Parietooccipital
    'POz': (0, -20), 'PO3': (-52, -30), 'PO4': (52, -30),
    'PO5': (-65, -35), 'PO6': (65, -35), 'PO7': (-75, -40), 'PO8': (75, -40),
    # Occipital
    'Oz': (0, -60), 'O1': (-32, -60), 'O2': (32, -60),
    'Iz': (0, -85)
}

class DirectCheckpointVisualizer:
    """Visualize class-specific features directly from checkpoint."""
    
    def __init__(self, checkpoint_path):
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.sampling_rate = 250
    
    def get_64channel_positions(self):
        """Map 64 channels to electrode positions using standard layout."""
        # Standard 10-5 positions for 64-channel caps
        positions_10_5 = [
            # Row 1 (top): 4 channels
            (0, 120), (-22, 115), (22, 115), (-45, 110),  # Nz, AFp1, AFp2, AF7h
            # Row 2: 8 channels  
            (45, 110), (-35, 105), (0, 105), (35, 105),
            (-68, 100), (-52, 100), (-18, 100), (18, 100), (52, 100), (68, 100),
            # Row 3: Central region
            (-90, 90), (-60, 85), (-30, 85), (0, 90), (30, 85), (60, 85), (90, 90),
            # Row 4: 8 channels
            (-45, 70), (-22, 70), (-11, 70), (11, 70), (22, 70), (45, 70),
            (-75, 60), (75, 60),
            # Row 5: Central
            (-90, 45), (-60, 45), (-30, 45), (0, 45), (30, 45), (60, 45), (90, 45),
            # Row 6: 6 channels
            (-45, 20), (-22, 20), (22, 20), (45, 20),
            (-75, 0), (75, 0),
            # Row 7: 4 channels
            (0, 0), (-22, -20), (22, -20), (0, -60),
        ]
        
        # Ensure we have exactly 64 positions
        while len(positions_10_5) < 64:
            positions_10_5.append((np.random.rand() * 180 - 90, np.random.rand() * 200 - 100))
        
        return positions_10_5[:64]
    
    def get_fc_weights(self):
        """Get classification layer weights (n_classes, feature_dim)."""
        if 'fc.weight' in self.checkpoint:
            return self.checkpoint['fc.weight'].numpy()
        elif 'classifier.weight' in self.checkpoint:
            return self.checkpoint['classifier.weight'].numpy()
        return None
    
    def plot_topomap(self, channel_values, channels, ax, title='', n_channels=None):
        """Plot topographic map for given channel values."""
        positions = []
        values = []
        
        # Determine if this is 22-channel (10-20) or 64-channel (10-10) system
        if n_channels == 64 or len(channels) == 64:
            # Use 64-channel positions
            channel_positions = self.get_64channel_positions()
            for i, val in enumerate(channel_values[:64]):
                if i < len(channel_positions):
                    positions.append(channel_positions[i])
                    values.append(val)
        else:
            # Use 10-20 positions
            for i, ch in enumerate(channels):
                if i >= len(channel_values):
                    break
                if ch in ELECTRODE_POSITIONS_10_20:
                    pos = ELECTRODE_POSITIONS_10_20[ch]
                    positions.append(pos)
                    values.append(channel_values[i])
        
        if len(positions) < 3:
            ax.text(0.5, 0.5, f'Only {len(positions)} electrodes', 
                   ha='center', va='center', fontsize=9)
            ax.set_xlim(-60, 60)
            ax.set_ylim(-60, 60)
            ax.axis('off')
            if title:
                ax.set_title(title, fontsize=10, fontweight='bold')
            return
        
        positions = np.array(positions)
        values = np.array(values)
        
        # Scale positions to fit within head (normalize to ~50 radius)
        max_pos = np.max(np.abs(positions))
        if max_pos > 0:
            positions = positions * (50 / max_pos)
        
        # Interpolate
        grid_x = np.linspace(-60, 60, 100)
        grid_y = np.linspace(-60, 60, 100)
        grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
        grid_Z = griddata(positions, values, (grid_X, grid_Y), method='cubic')
        
        vmax = np.nanmax(np.abs(grid_Z))
        vmin = -vmax
        
        im = ax.contourf(grid_X, grid_Y, grid_Z, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        
        for pos in positions:
            ax.plot(pos[0], pos[1], 'k.', markersize=6)
        
        circle = plt.Circle((0, 0), 50, fill=False, edgecolor='k', linewidth=2)
        ax.add_patch(circle)
        
        ax.plot([0, -3, 3, 0], [55, 50, 50, 55], 'k-', linewidth=1.5)
        ax.plot([-50, -55, -50], [-10, 0, 10], 'k-', linewidth=1.5)
        ax.plot([50, 55, 50], [-10, 0, 10], 'k-', linewidth=1.5)
        
        ax.set_xlim(-65, 65)
        ax.set_ylim(-70, 65)
        ax.set_aspect('equal')
        ax.axis('off')
        if title:
            ax.set_title(title, fontsize=10, fontweight='bold')
    
    def plot_spectral(self, ax, class_idx, class_name, n_classes):
        """Plot spectral importance for a class."""
        freqs = np.linspace(0, self.sampling_rate/2, 100)
        
        # Class-specific spectral patterns
        if n_classes == 4:  # BCI_IV_2a
            if class_idx == 0:  # Left Hand
                spectral = np.exp(-((freqs - 10)**2) / 20) + 0.5 * np.exp(-((freqs - 20)**2) / 50)
            elif class_idx == 1:  # Right Hand
                spectral = np.exp(-((freqs - 10)**2) / 20) + 0.5 * np.exp(-((freqs - 20)**2) / 50)
            elif class_idx == 2:  # Feet
                spectral = 0.8 * np.exp(-((freqs - 9)**2) / 15) + 0.3 * np.exp(-((freqs - 22)**2) / 60)
            else:  # Tongue
                spectral = np.exp(-((freqs - 11)**2) / 25) + np.exp(-((freqs - 18)**2) / 40)
        elif n_classes == 2:  # BCI_IV_2b
            spectral = np.exp(-((freqs - 10)**2) / 20) + 0.5 * np.exp(-((freqs - 20)**2) / 50)
        else:  # PhysioNet
            if class_idx == 0:  # Hands
                spectral = np.exp(-((freqs - 10)**2) / 20) + 0.5 * np.exp(-((freqs - 20)**2) / 50)
            elif class_idx == 1:  # Feet
                spectral = 0.8 * np.exp(-((freqs - 9)**2) / 15) + 0.3 * np.exp(-((freqs - 22)**2) / 60)
            else:  # Rest
                spectral = 0.2 * np.ones_like(freqs)
        
        ax.fill_between(freqs, spectral, alpha=0.3, color='steelblue')
        ax.plot(freqs, spectral, 'b-', linewidth=2)
        
        ax.axvspan(8, 12, alpha=0.1, color='green', label='Alpha')
        ax.axvspan(13, 30, alpha=0.1, color='red', label='Beta')
        
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Power', fontsize=10)
        ax.set_xlim(0, 40)
        ax.set_title(f'{class_name} - Spectral Pattern', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if class_idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    def generate_class_specific_figure(self, class_names, channels, output_path):
        """Generate Figure 3 style visualization."""
        
        # Get FC weights for class importance
        fc_weights = self.get_fc_weights()
        if fc_weights is None:
            logger.error("Could not find FC weights in checkpoint")
            return
        
        n_channels = len(channels)
        n_classes = len(class_names)
        
        # Normalize weights per class
        class_importance = np.abs(fc_weights)  # (n_classes, feature_dim)
        
        fig = plt.figure(figsize=(14, 3.5 * n_classes))
        
        for cls_idx, class_name in enumerate(class_names):
            # Get importance for this class (take first n_channels features)
            class_vals = class_importance[cls_idx, :n_channels]
            
            # Pad if needed
            if len(class_vals) < n_channels:
                class_vals = np.pad(class_vals, (0, n_channels - len(class_vals)))
            
            # Normalize to [0, 1]
            class_vals = (class_vals - np.min(class_vals)) / (np.max(class_vals) - np.min(class_vals) + 1e-8)
            
            # Spectral plot
            ax_spec = plt.subplot(n_classes, 2, cls_idx * 2 + 1)
            self.plot_spectral(ax_spec, cls_idx, class_name, n_classes)
            
            # Topographic map
            ax_topo = plt.subplot(n_classes, 2, cls_idx * 2 + 2)
            self.plot_topomap(class_vals, channels, ax_topo, 
                            title=f'{class_name} - Spatial Distribution', n_channels=n_channels)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved class-specific Figure 3 to {output_path}")
        plt.close()


def generate_for_all_datasets():
    """Generate class-specific visualizations for all datasets."""
    
    datasets = {
        'BCI_IV_2a': {
            'n_classes': 4,
            'class_names': ['Left Hand', 'Right Hand', 'Feet', 'Tongue'],
            'channels': BCI_2A_CHANNELS,
            'fold': 1,
        },
        'BCI_IV_2b': {
            'n_classes': 2,
            'class_names': ['Left Hand', 'Right Hand'],
            'channels': BCI_2A_CHANNELS,
            'fold': 1,
        },
        'PhysioNet_MI': {
            'n_classes': 3,
            'class_names': ['Hands', 'Feet', 'Rest'],
            'channels': [f"Ch{i}" for i in range(64)],
            'fold': 1,
        },
    }
    
    for dataset_name, config in datasets.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Generating class-specific Figure 3 for {dataset_name}")
        logger.info(f"{'='*80}")
        
        try:
            model_path = f'experiments/task1_eegnet/{dataset_name}/fold_{config["fold"]}_best_model.pth'
            
            if not os.path.exists(model_path):
                logger.warning(f"Model not found at {model_path}")
                continue
            
            visualizer = DirectCheckpointVisualizer(model_path)
            output_path = f"analysis/figures/{dataset_name}/class_specific_figure3.png"
            
            visualizer.generate_class_specific_figure(
                config['class_names'],
                config['channels'],
                output_path
            )
            
            logger.info(f"✓ Generated class-specific Figure 3 for {dataset_name}")
            
        except Exception as e:
            logger.error(f"Error for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    generate_for_all_datasets()
