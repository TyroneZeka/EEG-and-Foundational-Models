"""
Data loading module for EEG datasets.
Handles BCI_IV_2a (MOABB), BCI_IV_2b (MOABB), and PhysioNet MI (MNE).
"""

import os
import pickle
import numpy as np
import mne
from mne.datasets import eegbci
from moabb.datasets import BNCI2014001, BNCI2014004
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
    
    def load_bci_2b(self):
        """Load BCI_IV_2b dataset via MOABB."""
        logger.info("Loading BCI_IV_2b (BNCI2014004)...")
        dataset = BNCI2014004()
        
        # Use MotorImagery paradigm to load data
        paradigm = MotorImagery(n_classes=2)
        X, y, metadata = paradigm.get_data(dataset=dataset)
        
        logger.info(f"BCI_IV_2b: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Classes: {np.unique(y)}, Subjects: {len(np.unique(metadata['subject']))}")
        
        self.datasets['BCI_IV_2b'] = {
            'X': X,
            'y': y,
            'metadata': metadata,
            'channels': 3,
            'sampling_rate': 250,
            'n_subjects': len(np.unique(metadata['subject']))
        }
        return self.datasets['BCI_IV_2b']
    
    def load_physionet_mi(self):
        """Load PhysioNet MI dataset via MNE."""
        logger.info("Loading PhysioNet MI (eegbci)...")
        
        # Download PhysioNet MI data
        runs = [4, 8, 12]  # Motor imagery runs
        raw_fnames = eegbci.load_data(1, runs, update_path=True)
        
        X_list = []
        y_list = []
        subject_list = []
        
        for run_idx, fname in enumerate(raw_fnames):
            raw = mne.io.read_raw_edf(fname, preload=False)
            raw.load_data()
            
            # Pick EEG channels (exclude EOG and EMG)
            picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
            raw_eeg = raw.pick(picks)
            
            # Get events
            events = mne.events_from_annotations(raw)[0]
            event_id = {'T0': 0, 'T1': 1, 'T2': 2}  # rest, left hand, right hand
            
            # Create epochs
            epochs = mne.Epochs(
                raw_eeg, 
                events, 
                event_id=event_id,
                tmin=0, 
                tmax=4,
                baseline=None,
                preload=True
            )
            
            X_list.append(epochs.get_data())
            y_list.append(epochs.events[:, 2])
            subject_list.extend([1] * len(epochs))
        
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        
        logger.info(f"PhysioNet MI: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Classes: {np.unique(y)}")
        
        self.datasets['PhysioNet_MI'] = {
            'X': X,
            'y': y,
            'metadata': {'subject': np.array(subject_list)},
            'channels': X.shape[1],
            'sampling_rate': 160,
            'n_subjects': len(np.unique(subject_list))
        }
        return self.datasets['PhysioNet_MI']
    
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
    print(f"Window shape (Channels, Time samples): {window_shape_tuple}")
    print(f"Sampling frequency: {sfreq} Hz")
    print(f"Classes (integer labels): {targets}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message="Preprocessing choices with lambda functions cannot be saved.")
    load_data_final()
