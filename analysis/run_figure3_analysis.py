#!/usr/bin/env python
"""
Standalone Figure 3 Analysis Script
Generates visualizations from pre-trained models or current test data
Can be run independently after LOSO CV training completes
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.eegnet import EEGNet
from data.load_data import DataLoader as EEGDataLoader
from data.preprocessing import EEGPreprocessor
from analysis.interpretability_analyzer import InterpretabilityAnalyzer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_figure3_analysis_standalone():
    """
    Standalone analysis that generates Figure 3 visualizations.
    Works with test data loaded fresh from datasets.
    """
    
    logger.info("\n" + "="*80)
    logger.info("FIGURE 3 ANALYSIS - MODEL INTERPRETABILITY")
    logger.info("="*80 + "\n")
    
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
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Analyzing: {dataset_name}")
        logger.info(f"{'='*80}\n")
        
        try:
            # Load dataset
            load_func = getattr(data_loader, config['loader_func'])
            data = load_func()
            X, y = data['X'], data['y']
            metadata = data.get('metadata', None)
            
            logger.info(f"Loaded {X.shape[0]} samples, {X.shape[1]} channels, {len(np.unique(y))} classes")
            
            # Preprocess
            X = preprocessor.apply_average_reference(X)
            X = preprocessor.apply_bandpass_filter(X)
            
            # Convert labels if needed
            if y.dtype.kind in ('U', 'S', 'O'):
                unique_labels = np.unique(y)
                label_map = {label: idx for idx, label in enumerate(unique_labels)}
                y = np.array([label_map[label] for label in y], dtype=np.int64)
            
            n_samples = X.shape[2]
            
            # Create a trained model (or use pre-trained if available)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = EEGNet(n_channels, n_classes, n_samples, F1=16, F2=32, D=4).to(device)
            
            # Try to load a pre-trained model if available
            model_dir = f'experiments/task1_eegnet/{dataset_name}'
            if os.path.exists(model_dir):
                # Use the first fold's model as representative
                model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pth')])
                if model_files:
                    model_path = os.path.join(model_dir, model_files[0])
                    logger.info(f"Loading pre-trained model from {model_path}...")
                    model.load_state_dict(torch.load(model_path, map_location=device))
                else:
                    logger.warning(f"No model files found in {model_dir}, using untrained model")
            else:
                logger.warning(f"Model directory {model_dir} not found, using untrained model for visualization demo")
            
            model.eval()
            
            # Create analyzer
            analyzer = InterpretabilityAnalyzer(
                model=model,
                dataset_name=dataset_name,
                n_channels=n_channels,
                n_classes=n_classes,
                sampling_rate=250
            )
            
            # Run analysis on aggregate data
            dataset_output_dir = os.path.join(output_base_dir, dataset_name)
            analyzer.analyze_aggregate(X, y, dataset_output_dir)
            
            logger.info(f"✓ Analysis complete for {dataset_name}\n")
            
        except Exception as e:
            logger.error(f"Failed to analyze {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("\n" + "="*80)
    logger.info("FIGURE 3 ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {output_base_dir}")
    logger.info("="*80 + "\n")
    
    # Summary
    logger.info("\nGenerated visualizations:")
    for dataset in ['BCI_IV_2a', 'BCI_IV_2b', 'PhysioNet_MI']:
        dataset_dir = os.path.join(output_base_dir, dataset)
        if os.path.exists(dataset_dir):
            files = os.listdir(dataset_dir)
            logger.info(f"\n{dataset}:")
            for f in sorted(files):
                logger.info(f"  - {f}")


if __name__ == "__main__":
    run_figure3_analysis_standalone()
