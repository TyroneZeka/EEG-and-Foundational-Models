import os
import sys
import numpy as np
from data.load_data import DataLoader as EEGDataLoader
from data.preprocessing import preprocess_dataset
from model.train_eegnet import loso_cross_validation

# Add project to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_all_datasets():
    """Train EEGNet on all three datasets."""
    
    loader = EEGDataLoader()
    
    # Dataset configurations
    datasets = {
        'BCI_IV_2b': {'loader_func': loader.load_bci_2b, 'sampling_rate': 250},
        'PhysioNet_MI': {'loader_func': loader.load_physionet_mi, 'sampling_rate': 160},
    }
    
    results_summary = {}
    
    for dataset_name, config in datasets.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING ON {dataset_name}")
        logger.info(f"{'='*80}")
        
        try:
            # Load dataset
            logger.info(f"Loading {dataset_name}...")
            data = config['loader_func']()
            
            # Preprocess
            logger.info(f"Preprocessing {dataset_name}...")
            data = preprocess_dataset(data, sampling_rate=config['sampling_rate'])
            
            # Run LOSO CV
            logger.info(f"Starting LOSO CV on {dataset_name}...")
            fold_results = loso_cross_validation(
                data['X'],
                data['y'],
                data['metadata'],
                dataset_name=dataset_name,
                log_dir='logs/task1_eegnet'
            )
            
            # Aggregate results
            test_accs = [r['test_balanced_acc'] for r in fold_results]
            mean_acc = np.mean(test_accs)
            std_acc = np.std(test_accs)
            
            results_summary[dataset_name] = {
                'mean': mean_acc,
                'std': std_acc,
                'per_fold': test_accs
            }
            
            logger.info(f"\n{'='*80}")
            logger.info(f"{dataset_name} SUMMARY")
            logger.info(f"{'='*80}")
            logger.info(f"Mean Balanced Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
            logger.info(f"Per-fold accuracies: {[f'{acc:.4f}' for acc in test_accs]}")
            
        except Exception as e:
            logger.error(f"Error training on {dataset_name}: {e}")
            results_summary[dataset_name] = {'error': str(e)}
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("FINAL SUMMARY - ALL DATASETS")
    logger.info(f"{'='*80}")
    
    for dataset_name, result in results_summary.items():
        if 'error' not in result:
            logger.info(f"{dataset_name}: {result['mean']:.4f} +/- {result['std']:.4f}")
        else:
            logger.error(f"{dataset_name}: ERROR - {result['error']}")


if __name__ == "__main__":
    train_all_datasets()