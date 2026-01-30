"""
Data loading module for EEG datasets.
Handles BCI_IV_2a (MOABB), BCI_IV_2b (MOABB), and PhysioNet MI (MNE).
"""

import os
import pickle
import numpy as np
import mne
from mne.datasets import eegbci
import moabb
from moabb.datasets import BNCI2014001,Lee2019_MI, BNCI2015_001
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and cache EEG datasets."""
    
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.datasets = {}
    
    def load_bci_2a(self):
        """Load BCI_IV_2a dataset via MOABB."""
        logger.info("Loading BCI_IV_2a (BNCI2014001)...")
        dataset = BNCI2014001()
        
        # Use MotorImagery paradigm to load data
        paradigm = MotorImagery(n_classes=4)
        X, y, metadata = paradigm.get_data(dataset=dataset)
        
        logger.info(f"BCI_IV_2a: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Classes: {np.unique(y)}, Subjects: {len(np.unique(metadata['subject']))}")
        
        self.datasets['BCI_IV_2a'] = {
            'X': X,
            'y': y,
            'metadata': metadata,
            'channels': 22,
            'sampling_rate': 250,
            'n_subjects': len(np.unique(metadata['subject']))
        }
        return self.datasets['BCI_IV_2a']
    
    
    def load_lee2019(self):
        """Load Lee2019 dataset via MOABB."""
        logger.info("Loading Lee2019 MI dataset...")
        dataset = Lee2019_MI()
        paradigm = MotorImagery(n_classes=2)
        X, y, metadata = paradigm.get_data(dataset=dataset)

        logger.info(f"Lee2019: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Classes: {np.unique(y)}, Subjects: {len(np.unique(metadata['subject']))}")

        self.datasets['Lee2019'] = {
            'X': X, 'y': y, 'metadata': metadata,
            'channels': X.shape[1],
            'sampling_rate': 500,
            'n_subjects': len(np.unique(metadata['subject']))
        }
        return self.datasets['Lee2019']

    def load_bnci2015_001(self):
        """Load BNCI2015-001 Motor Imagery dataset via MOABB."""
        logger.info("Loading BNCI2015-001 MI dataset...")
        dataset = BNCI2015_001()

        # The dataset description says it's right hand vs. feet, which is a 2-class MI task
        paradigm = MotorImagery(n_classes=2) 

        # Use the paradigm to get the data. MOABB will handle the sessions correctly.
        X, y, metadata = paradigm.get_data(dataset=dataset)
        # --- END OF FIX ---

        logger.info(f"BNCI2015-001: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Classes: {np.unique(y)}, Subjects: {len(np.unique(metadata['subject']))}")

        self.datasets['BNCI2015_001'] = {
            'X': X,
            'y': y,
            'metadata': metadata,
            'channels': X.shape[1],
            'sampling_rate': 512, # From the class description
            'n_subjects': len(np.unique(metadata['subject']))
        }
        return self.datasets['BNCI2015_001']
    

    def load_all(self):
        """Load all three datasets."""
        logger.info("Starting data loading for all datasets...")
        
        try:
            self.load_bci_2a()
        except Exception as e:
            logger.error(f"Error loading BCI_IV_2a: {e}")
        
        try:
            self.load_bci_2b()
        except Exception as e:
            logger.error(f"Error loading BCI_IV_2b: {e}")
        
        try:
            self.load_physionet_mi()
        except Exception as e:
            logger.error(f"Error loading PhysioNet MI: {e}")
        
        return self.datasets
    
    def save_summary(self, output_file="data/dataset_summary.txt"):
        """Save dataset summary to file."""
        with open(output_file, 'w') as f:
            f.write("EEG DATASETS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            for name, data in self.datasets.items():
                f.write(f"Dataset: {name}\n")
                f.write(f"  Shape: {data['X'].shape}\n")
                f.write(f"  Channels: {data['channels']}\n")
                f.write(f"  Sampling Rate: {data['sampling_rate']} Hz\n")
                f.write(f"  Subjects: {data['n_subjects']}\n")
                f.write(f"  Classes: {np.unique(data['y'])}\n")
                f.write(f"  Class distribution: {np.bincount(data['y'])}\n")
                f.write("\n")
        
        logger.info(f"Summary saved to {output_file}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message="Preprocessing choices with lambda functions cannot be saved.")
    # Simple demo: load all datasets and write summary
    loader = DataLoader()
    loader.load_all()
    loader.save_summary()
