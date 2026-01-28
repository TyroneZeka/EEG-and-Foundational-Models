#!/usr/bin/env python
"""Diagnostic script to check data quality and model performance."""

import os
import sys
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_data import DataLoader
from data.preprocessing import preprocess_dataset
from models.eegnet import EEGNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_data():
    """Check data quality and distributions."""
    logger.info("="*80)
    logger.info("DATA QUALITY DIAGNOSTICS")
    logger.info("="*80)
    
    loader = DataLoader()
    data_2a = loader.load_bci_2a()
    
    X_raw = data_2a['X']
    y_raw = data_2a['y']
    
    # Check 1: Raw data statistics
    logger.info("\n[1] RAW DATA STATISTICS")
    logger.info(f"  Shape: {X_raw.shape}")
    logger.info(f"  Data type: {X_raw.dtype}")
    logger.info(f"  NaN values: {np.isnan(X_raw).sum()}")
    logger.info(f"  Inf values: {np.isinf(X_raw).sum()}")
    logger.info(f"  Min: {np.min(X_raw):.6f}, Max: {np.max(X_raw):.6f}")
    logger.info(f"  Mean: {np.mean(X_raw):.6f}, Std: {np.std(X_raw):.6f}")
    
    # Check 2: Label distribution
    logger.info("\n[2] LABEL DISTRIBUTION (RAW)")
    logger.info(f"  Label dtype: {y_raw.dtype}")
    logger.info(f"  Unique labels: {np.unique(y_raw)}")
    unique, counts = np.unique(y_raw, return_counts=True)
    for label, count in zip(unique, counts):
        logger.info(f"    Class {label}: {count} samples ({100*count/len(y_raw):.1f}%)")
    
    # Check 3: Preprocess
    logger.info("\n[3] PREPROCESSING")
    data_2a_prep = preprocess_dataset(data_2a, sampling_rate=250)
    X_prep = data_2a_prep['X']
    y_prep = data_2a_prep['y']
    
    logger.info(f"  Preprocessed shape: {X_prep.shape}")
    logger.info(f"  Preprocessed dtype: {X_prep.dtype}")
    logger.info(f"  NaN values: {np.isnan(X_prep).sum()}")
    logger.info(f"  Inf values: {np.isinf(X_prep).sum()}")
    logger.info(f"  Min: {np.min(X_prep):.6f}, Max: {np.max(X_prep):.6f}")
    logger.info(f"  Mean: {np.mean(X_prep):.6f}, Std: {np.std(X_prep):.6f}")
    
    # Check 4: Label preservation
    logger.info("\n[4] LABEL PRESERVATION")
    logger.info(f"  Original labels dtype: {y_raw.dtype}")
    logger.info(f"  Preprocessed labels dtype: {y_prep.dtype}")
    logger.info(f"  Labels changed: {not np.array_equal(np.sort(np.unique(y_raw)), np.sort(np.unique(y_prep)))}")
    
    unique_prep, counts_prep = np.unique(y_prep, return_counts=True)
    for label, count in zip(unique_prep, counts_prep):
        logger.info(f"    Class {label}: {count} samples ({100*count/len(y_prep):.1f}%)")
    
    # Check 5: Single sample check
    logger.info("\n[5] SAMPLE INTEGRITY CHECK")
    sample_idx = 0
    logger.info(f"  Sample {sample_idx}:")
    logger.info(f"    Original: shape={X_raw[sample_idx].shape}, min={np.min(X_raw[sample_idx]):.6f}, max={np.max(X_raw[sample_idx]):.6f}")
    logger.info(f"    Preprocessed: shape={X_prep[sample_idx].shape}, min={np.min(X_prep[sample_idx]):.6f}, max={np.max(X_prep[sample_idx]):.6f}")
    logger.info(f"    Label: {y_raw[sample_idx]} -> {y_prep[sample_idx]}")
    
    # Check 6: Channel-wise statistics
    logger.info("\n[6] CHANNEL-WISE STATISTICS (Preprocessed)")
    for ch in range(min(3, X_prep.shape[1])):  # First 3 channels
        ch_data = X_prep[:, ch, :]
        logger.info(f"  Channel {ch}: min={np.min(ch_data):.4f}, max={np.max(ch_data):.4f}, mean={np.mean(ch_data):.4f}, std={np.std(ch_data):.4f}")
    
    return X_prep, y_prep

def diagnose_model(X, y):
    """Check model forward pass and output distribution."""
    logger.info("\n" + "="*80)
    logger.info("MODEL DIAGNOSTICS")
    logger.info("="*80)
    
    n_channels = X.shape[1]
    n_samples = X.shape[2]
    n_classes = len(np.unique(y))
    
    logger.info(f"\nModel Configuration:")
    logger.info(f"  Input shape: (batch, {n_channels}, {n_samples})")
    logger.info(f"  Output classes: {n_classes}")
    
    # Create model
    model = EEGNet(n_channels, n_classes, n_samples)
    model.eval()
    
    # Test forward pass
    X_test = torch.FloatTensor(X[:10])
    with torch.no_grad():
        logits = model(X_test)
    
    logger.info(f"\nForward Pass Test:")
    logger.info(f"  Input batch shape: {X_test.shape}")
    logger.info(f"  Output logits shape: {logits.shape}")
    logger.info(f"  Output dtype: {logits.dtype}")
    logger.info(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    
    # Check softmax
    probs = torch.softmax(logits, dim=1)
    logger.info(f"  Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
    logger.info(f"  Probability sum (should be ~1.0): {probs.sum(dim=1)[:3].tolist()}")
    
    # Check predictions
    preds = torch.argmax(logits, dim=1)
    logger.info(f"  Predicted classes: {np.unique(preds.numpy())}")
    logger.info(f"  ✓ Model forward pass OK")

if __name__ == "__main__":
    X, y = diagnose_data()
    diagnose_model(X, y)
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTICS COMPLETE")
    logger.info("="*80)
