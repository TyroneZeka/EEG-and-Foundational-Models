"""
t-SNE Analysis: Visualize learned feature representations
Shows how the model separates different motor imagery classes in learned feature space
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.eegnet import EEGNet
from data.load_data import DataLoader as EEGDataLoader
from data.preprocessing import EEGPreprocessor
from analysis.model_loader import get_eegnet_params_from_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract intermediate features from trained models."""
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.model.eval()
    
    def extract_features(self, X, layer_name='before_classifier'):
        """Extract features from specified layer."""
        X_torch = torch.FloatTensor(X).to(self.device)
        
        features_list = []
        
        with torch.no_grad():
            for batch_idx in tqdm(range(0, len(X), 32), desc="Extracting features"):
                batch = X_torch[batch_idx:batch_idx+32]
                
                # Forward pass and capture intermediate features
                x = batch
                for name, module in self.model.named_modules():
                    x = module(x)
                    
                    # Extract before final classification layer
                    if 'classifier' in name.lower() and isinstance(module, nn.Linear):
                        # Features are the output of the previous layer
                        break
                
                # Flatten if needed
                if len(x.shape) > 2:
                    x = x.view(x.shape[0], -1)
                
                features_list.append(x.cpu().numpy())
        
        features = np.vstack(features_list)
        return features
    
    def extract_before_classifier(self, X):
        """Extract features right before classification layer."""
        X_torch = torch.FloatTensor(X).to(self.device)
        
        features_list = []
        
        with torch.no_grad():
            for batch_idx in tqdm(range(0, len(X), 32), desc="Extracting pre-classifier features"):
                batch = X_torch[batch_idx:batch_idx+32]
                
                # Forward through model, capturing output before FC layer
                x = self.model.conv1(batch)
                x = self.model.batchnorm1(x)
                x = self.model.depthwise(x)
                x = self.model.batchnorm2(x)
                x = self.model.separable(x)
                x = self.model.batchnorm3(x)
                x = self.model.pool(x)
                x = self.model.dropout(x)
                x = self.model.pool2(x)
                x = self.model.dropout2(x)
                x = x.view(x.shape[0], -1)  # Flatten
                
                features_list.append(x.cpu().numpy())
        
        features = np.vstack(features_list)
        return features


class TSNEVisualizer:
    """Create t-SNE visualizations of learned features."""
    
    def __init__(self, perplexity=30, n_iter=1000, random_state=42):
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit_tsne(self, features, labels):
        """Fit t-SNE and return 2D projections."""
        logger.info(f"Computing t-SNE (perplexity={self.perplexity}, n_iter={self.n_iter})...")
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Fit t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=self.perplexity,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=1
        )
        projections = tsne.fit_transform(features_scaled)
        
        return projections
    
    def plot_tsne(self, projections, labels, class_names, output_path):
        """Create t-SNE visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
        
        # Plot 1: By class
        ax = axes[0]
        for class_idx, class_name in enumerate(class_names):
            mask = labels == class_idx
            ax.scatter(
                projections[mask, 0],
                projections[mask, 1],
                c=[colors[class_idx]],
                label=class_name,
                s=50,
                alpha=0.7,
                edgecolors='k',
                linewidth=0.5
            )
        
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.set_title('Learned Feature Space (Colored by Class)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Density/cluster quality
        ax = axes[1]
        scatter = ax.scatter(
            projections[:, 0],
            projections[:, 1],
            c=labels,
            cmap='tab10',
            s=50,
            alpha=0.7,
            edgecolors='k',
            linewidth=0.5
        )
        
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.set_title('Feature Space Density', fontsize=13, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Class Index', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved t-SNE visualization to {output_path}")
        plt.close()
        
        return projections


def analyze_all_datasets():
    """Generate t-SNE visualizations for all datasets."""
    
    datasets = {
        'BCI_IV_2a': {
            'loader': 'load_bci_2a',
            'n_channels': 22,
            'n_classes': 4,
            'class_names': ['Left Hand', 'Right Hand', 'Feet', 'Tongue'],
            'n_samples': 1000,
            'sampling_rate': 250,
            'fold': 1,
            'F1': 16, 'F2': 32, 'D': 4,
        },
        'BCI_IV_2b': {
            'loader': 'load_bci_2b',
            'n_channels': 22,
            'n_classes': 2,
            'class_names': ['Left Hand', 'Right Hand'],
            'n_samples': 1000,
            'sampling_rate': 250,
            'fold': 1,
            'F1': 16, 'F2': 32, 'D': 4,
        },
        'PhysioNet_MI': {
            'loader': 'load_physionet_mi',
            'n_channels': 64,
            'n_classes': 3,
            'class_names': ['Hands', 'Feet', 'Rest'],
            'n_samples': 1600,
            'sampling_rate': 160,
            'fold': 1,
            'F1': 16, 'F2': 32, 'D': 4,
        },
    }
    
    data_loader = EEGDataLoader()
    preprocessor = EEGPreprocessor(sampling_rate=250)
    visualizer = TSNEVisualizer(perplexity=30, n_iter=1000)
    
    for dataset_name, config in datasets.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"t-SNE Analysis: {dataset_name}")
        logger.info(f"{'='*80}")
        
        try:
            # Load data
            loader_func = getattr(data_loader, config['loader'])
            data = loader_func()
            X, y = data['X'], data['y']
            
            logger.info(f"Loaded data: X shape={X.shape}, y shape={y.shape}")
            logger.info(f"Classes: {np.unique(y)}")
            
            # Load model
            model_path = f'experiments/task1_eegnet/{dataset_name}/fold_{config["fold"]}_best_model.pth'
            
            if not os.path.exists(model_path):
                logger.warning(f"Model not found at {model_path}")
                continue
            
            # Infer architecture from checkpoint
            params = get_eegnet_params_from_checkpoint(model_path)
            # Create model with correct n_samples from data
            n_samples = X.shape[2]  # Get from actual data
            model = EEGNet(
                n_channels=params['n_channels'],
                n_classes=params['n_classes'],
                n_samples=n_samples,
                F1=params['F1'],
                F2=params['F2'],
                D=params['D']
            )
            model.load_state_dict(torch.load(model_path))
            model.eval()
            logger.info("✓ Model loaded")
            
            # Extract features
            extractor = FeatureExtractor(model)
            features = extractor.extract_before_classifier(X)
            logger.info(f"Extracted features: shape={features.shape}")
            
            # Fit t-SNE
            projections = visualizer.fit_tsne(features, y)
            
            # Create visualization
            output_path = f"analysis/figures/{dataset_name}/tsne_visualization.png"
            visualizer.plot_tsne(projections, y, config['class_names'], output_path)
            
            # Additional stats
            logger.info(f"\nFeature space analysis:")
            logger.info(f"  - Feature dimension: {features.shape[1]}")
            logger.info(f"  - Total samples: {len(X)}")
            for cls_idx, class_name in enumerate(config['class_names']):
                n_cls = np.sum(y == cls_idx)
                logger.info(f"  - {class_name}: {n_cls} samples")
            
        except Exception as e:
            logger.error(f"Error analyzing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    analyze_all_datasets()
