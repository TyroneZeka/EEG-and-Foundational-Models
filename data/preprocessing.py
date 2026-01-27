"""
Preprocessing module for EEG data.
Implements: average reference, bandpass filter, epoching, z-score normalization.
"""

import numpy as np
from scipy import signal
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """Unified preprocessing pipeline for EEG data."""
    
    def __init__(self, 
                 sampling_rate,
                 bandpass_hz=[4, 38],
                 reference='average',
                 norm_type='zscore',
                 random_seed=42):
        """
        Initialize preprocessor.
        
        Args:
            sampling_rate: Sampling rate in Hz
            bandpass_hz: Tuple of (low_freq, high_freq) for bandpass filter
            reference: 'average' or 'csd' (Common Spatial Derivation)
            norm_type: 'zscore' or 'minmax'
            random_seed: For reproducibility
        """
        self.sampling_rate = sampling_rate
        self.bandpass_hz = bandpass_hz
        self.reference = reference
        self.norm_type = norm_type
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def apply_average_reference(self, X):
        """Apply average reference: X = X - mean(X)."""
        logger.info("Applying average reference...")
        # X shape: (n_samples, n_channels, n_times)
        mean_ref = np.mean(X, axis=1, keepdims=True)
        X_ref = X - mean_ref
        return X_ref
    
    def apply_bandpass_filter(self, X):
        """Apply zero-phase bandpass filter (4-38 Hz)."""
        logger.info(f"Applying bandpass filter {self.bandpass_hz[0]}-{self.bandpass_hz[1]} Hz...")
        nyquist = self.sampling_rate / 2
        low = self.bandpass_hz[0] / nyquist
        high = self.bandpass_hz[1] / nyquist
        
        # Design filter
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        
        # Apply filter (zero-phase)
        X_filtered = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_filtered[i, j, :] = signal.sosfiltfilt(sos, X[i, j, :])
        
        return X_filtered
    
    def normalize_zscore(self, X_train, X_val=None, X_test=None):
        """Z-score normalization: fit on training data only."""
        logger.info("Applying z-score normalization (fit on training)...")
        
        # Compute mean and std from training data only
        mean = np.mean(X_train, axis=(0, 2), keepdims=True)
        std = np.std(X_train, axis=(0, 2), keepdims=True)
        std[std == 0] = 1  # Avoid division by zero
        
        X_train_norm = (X_train - mean) / std
        
        results = {'train': X_train_norm}
        if X_val is not None:
            results['val'] = (X_val - mean) / std
        if X_test is not None:
            results['test'] = (X_test - mean) / std
        
        return results
    
    def preprocess(self, X, y=None, split_indices=None):
        """
        Full preprocessing pipeline.
        
        Args:
            X: (n_samples, n_channels, n_times)
            y: (n_samples,) optional labels
            split_indices: dict with 'train', 'val', 'test' indices for stratified normalization
        
        Returns:
            X_processed: preprocessed data
            y: labels (converted to int if needed)
        """
        logger.info(f"Starting preprocessing. Input shape: {X.shape}")
        
        # Convert labels to integers if they are strings (MOABB paradigm output)
        if y is not None:
            if y.dtype.kind in ('U', 'S', 'O'):  # Unicode, byte string, or object
                logger.info("Converting string labels to integers...")
                unique_labels = np.unique(y)
                label_map = {label: idx for idx, label in enumerate(unique_labels)}
                y = np.array([label_map[label] for label in y], dtype=np.int64)
                logger.info(f"Label mapping: {label_map}")
            else:
                y = y.astype(np.int64)
        
        # Step 1: Average reference
        X = self.apply_average_reference(X)
        
        # Step 2: Bandpass filter
        X = self.apply_bandpass_filter(X)
        
        # Step 3: Z-score normalization
        if split_indices is not None:
            X_train = X[split_indices['train']]
            X_val = X[split_indices['val']] if 'val' in split_indices else None
            X_test = X[split_indices['test']] if 'test' in split_indices else None
            
            norm_results = self.normalize_zscore(X_train, X_val, X_test)
            X = np.zeros_like(X)
            X[split_indices['train']] = norm_results['train']
            if 'val' in split_indices:
                X[split_indices['val']] = norm_results['val']
            if 'test' in split_indices:
                X[split_indices['test']] = norm_results['test']
        else:
            norm_results = self.normalize_zscore(X)
            X = norm_results['train']
        
        logger.info(f"Preprocessing complete. Output shape: {X.shape}")
        return X, y
    
    def get_temporal_statistics(self, X):
        """Get temporal statistics for each sample."""
        stats = {
            'mean': np.mean(X, axis=(1, 2)),
            'std': np.std(X, axis=(1, 2)),
            'min': np.min(X, axis=(1, 2)),
            'max': np.max(X, axis=(1, 2))
        }
        return stats


def preprocess_dataset(data_dict, sampling_rate, bandpass_hz=[4, 38], random_seed=42):
    """
    Preprocess entire dataset with stratified train/val/test split normalization.
    
    Args:
        data_dict: dict with 'X', 'y', 'metadata'
        sampling_rate: Sampling rate in Hz
        bandpass_hz: Bandpass filter range
        random_seed: For reproducibility
    
    Returns:
        Preprocessed data_dict
    """
    preprocessor = EEGPreprocessor(
        sampling_rate=sampling_rate,
        bandpass_hz=bandpass_hz,
        random_seed=random_seed
    )
    
    X, y = preprocessor.preprocess(data_dict['X'], data_dict['y'])
    data_dict['X'] = X
    data_dict['y'] = y
    
    return data_dict
