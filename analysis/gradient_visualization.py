"""
Gradient Visualization - Analyze gradient flow during training
Shows how gradients propagate through early, middle, and late training stages
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.eegnet import EEGNet
from data.load_data import DataLoader as EEGDataLoader
from data.preprocessing import EEGPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradientVisualizer:
    """Visualize gradient flow through network layers."""
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.model.eval()
    
    def compute_gradients(self, X, y):
        """Compute gradients for a batch."""
        X_torch = torch.FloatTensor(X).to(self.device)
        y_torch = torch.LongTensor(y).to(self.device)
        
        self.model.train()
        
        # Forward pass
        logits = self.model(X_torch)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y_torch)
        
        # Backward pass
        if self.model.zero_grad is not None:
            self.model.zero_grad()
        loss.backward()
        
        # Collect gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.abs().detach().cpu().numpy()
        
        self.model.eval()
        return gradients
    
    def get_layer_gradient_stats(self, gradients):
        """Compute statistics for each layer."""
        stats = {}
        
        for layer_name, grad in gradients.items():
            stats[layer_name] = {
                'mean': float(np.mean(grad)),
                'std': float(np.std(grad)),
                'max': float(np.max(grad)),
                'min': float(np.min(grad)),
            }
        
        return stats


class GradientMonitor:
    """Monitor gradient flow during simulated training."""
    
    def __init__(self, model_path, dataset_name):
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.device = 'cpu'
        self._load_model()
    
    def _load_model(self):
        """Load trained model."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Infer architecture
        conv1_weight = checkpoint['conv1.weight']
        fc_weight = checkpoint.get('fc.weight', checkpoint.get('classifier.weight'))
        depthwise = checkpoint['depthwise.weight']
        
        n_channels = depthwise.shape[2]  # Correct: [F1*D, 1, n_channels, 1]
        n_classes = fc_weight.shape[0]
        F1 = conv1_weight.shape[0]
        
        # Create model
        self.model = EEGNet(
            n_channels=n_channels,
            n_classes=n_classes,
            n_samples=1000,
            F1=F1,
            F2=32,
            D=4
        )
        
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✓ Loaded model: {n_channels} channels, {n_classes} classes")
    
    def visualize_gradients(self, X, y, output_path):
        """Visualize gradient statistics across layers."""
        visualizer = GradientVisualizer(self.model)
        
        # Get sample batches
        batch_indices = np.random.choice(len(X), min(32, len(X)), replace=False)
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        # Compute gradients
        gradients = visualizer.compute_gradients(X_batch, y_batch)
        stats = visualizer.get_layer_gradient_stats(gradients)
        
        # Group by layer type
        layer_groups = {
            'Early (Conv1-BatchNorm1)': [],
            'Middle (Depthwise-Separable)': [],
            'Late (Pooling-FC)': []
        }
        
        for layer_name, layer_stats in stats.items():
            if 'conv1' in layer_name or 'batchnorm1' in layer_name:
                group = 'Early (Conv1-BatchNorm1)'
            elif 'depthwise' in layer_name or 'separable' in layer_name or 'batchnorm2' in layer_name or 'batchnorm3' in layer_name:
                group = 'Middle (Depthwise-Separable)'
            else:
                group = 'Late (Pooling-FC)'
            
            layer_groups[group].append({
                'name': layer_name,
                'stats': layer_stats
            })
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        for idx, (group_name, layers) in enumerate(layer_groups.items()):
            ax = axes[idx]
            
            layer_names = [l['name'].split('.')[-1][:15] for l in layers]
            means = [l['stats']['mean'] for l in layers]
            stds = [l['stats']['std'] for l in layers]
            
            x_pos = np.arange(len(layer_names))
            ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(layer_names, rotation=45, ha='right')
            ax.set_ylabel('Mean Gradient Magnitude')
            ax.set_title(f'{group_name}\nGradient Flow', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved to {output_path}")
        plt.close()
        
        return stats


def analyze_gradients(dataset_name):
    """Analyze gradient flow for dataset."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Gradient Analysis: {dataset_name}")
    logger.info(f"{'='*80}\n")
    
    # Load data
    loader = EEGDataLoader()
    if dataset_name == 'BCI_IV_2a':
        data = loader.load_bci_2a()
    elif dataset_name == 'BCI_IV_2b':
        data = loader.load_bci_2b()
    elif dataset_name == 'PhysioNet_MI':
        data = loader.load_physionet_mi()
    
    X, y = data['X'], data['y']
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Preprocess
    preprocessor = EEGPreprocessor(sampling_rate=250)
    X_processed, _ = preprocessor.preprocess(X, y)
    
    # Load model
    model_path = f"experiments/task1_eegnet/{dataset_name}/fold_1_best_model.pth"
    if not os.path.exists(model_path):
        logger.warning(f"Model not found: {model_path}")
        return
    
    try:
        monitor = GradientMonitor(model_path, dataset_name)
        output_path = f"analysis/figures/{dataset_name}/gradient_visualization.png"
        monitor.visualize_gradients(X_processed, y, output_path)
        logger.info(f"✓ Gradient visualization complete for {dataset_name}")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run gradient analysis for all datasets."""
    
    datasets = ['BCI_IV_2a', 'BCI_IV_2b', 'PhysioNet_MI']
    
    for dataset_name in datasets:
        try:
            analyze_gradients(dataset_name)
        except Exception as e:
            logger.error(f"Error analyzing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*80}")
    logger.info("Gradient Analysis Complete")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
