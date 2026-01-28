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
        """Load PhysioNet MI dataset via MNE from multiple subjects."""
        logger.info("Loading PhysioNet MI (eegbci) - Multiple Subjects...")
        
        # Load multiple subjects (1-5) with motor imagery runs
        subjects = list(range(1, 6))  # Load subjects 1-5
        runs = [4, 8, 12]  # Motor imagery runs (left hand, right hand, feet)
        
        X_list = []
        y_list = []
        subject_list = []
        
        for subject_id in subjects:
            logger.info(f"  Loading subject {subject_id}...")
            try:
                raw_fnames = eegbci.load_data(subject_id, runs, update_path=True)
            except Exception as e:
                logger.warning(f"Could not load subject {subject_id}: {e}, skipping...")
                continue
            
            for run_idx, fname in enumerate(raw_fnames):
                try:
                    raw = mne.io.read_raw_edf(fname, preload=False)
                    raw.load_data()

                    # Pick EEG channels (exclude EOG and EMG)
                    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
                    raw_eeg = raw.pick(picks)

                    # Extract events and mapping from annotations
                    events, ann_event_id = mne.events_from_annotations(raw)

                    # Select motor-imagery annotations (commonly 'T0','T1','T2') if present
                    desired_keys = [k for k in ann_event_id.keys() if k in ('T0', 'T1', 'T2')]
                    if not desired_keys:
                        # Fallback: use all annotation keys
                        desired_event_id = ann_event_id
                    else:
                        desired_event_id = {k: ann_event_id[k] for k in desired_keys}

                    # Create epochs using the selected event mapping
                    epochs = mne.Epochs(
                        raw_eeg,
                        events,
                        event_id=desired_event_id,
                        tmin=0,
                        tmax=4,
                        baseline=None,
                        preload=True,
                    )

                    # Get numeric event codes and remap to contiguous labels (0..C-1)
                    codes = epochs.events[:, 2]
                    unique_codes = np.unique(codes)
                    code2label = {code: idx for idx, code in enumerate(sorted(unique_codes))}
                    labels = np.array([code2label[c] for c in codes], dtype=np.int64)

                    X_list.append(epochs.get_data())
                    y_list.append(labels)
                    subject_list.extend([subject_id] * len(epochs))
                
                except Exception as e:
                    logger.warning(f"Error processing run {run_idx} for subject {subject_id}: {e}")
                    continue
        
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


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message="Preprocessing choices with lambda functions cannot be saved.")
    # Simple demo: load all datasets and write summary
    loader = DataLoader()
    loader.load_all()
    loader.save_summary()
