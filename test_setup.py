#!/usr/bin/env python
"""Quick test script to verify setup."""

import sys
sys.path.insert(0, '.')

print("Testing imports and models...")

# Test imports
try:
    import torch
    import numpy as np
    from models.eegnet import EEGNet
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test EEGNet
try:
    model = EEGNet(n_channels=22, n_classes=4, n_samples=1000)
    x = torch.randn(2, 22, 1000)
    out = model(x)
    assert out.shape == (2, 4), f"Expected shape (2, 4), got {out.shape}"
    print(f"✓ EEGNet forward pass OK: {x.shape} -> {out.shape}")
except Exception as e:
    print(f"✗ EEGNet test failed: {e}")
    sys.exit(1)

# Test preprocessing module
try:
    from data.preprocessing import EEGPreprocessor
    preprocessor = EEGPreprocessor(sampling_rate=250)
    X = np.random.randn(10, 22, 250)
    y = np.random.randint(0, 4, 10)
    X_proc, y_proc = preprocessor.preprocess(X, y)
    print(f"✓ Preprocessing pipeline OK: {X.shape} -> {X_proc.shape}")
except Exception as e:
    print(f"✗ Preprocessing test failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("All tests passed! System is ready.")
print("="*80)
