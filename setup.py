#!/usr/bin/env python
"""
Setup script: Install dependencies, download datasets, verify setup.
Run this first: python setup.py
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def install_dependencies():
    """Install required packages."""
    logger.info("Installing dependencies from requirements.txt...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    logger.info("Dependencies installed successfully!")


def verify_imports():
    """Verify that all required packages can be imported."""
    logger.info("Verifying imports...")
    required = ['torch', 'numpy', 'scipy', 'sklearn', 'mne', 'moabb', 'tensorboard']
    
    for package in required:
        try:
            __import__(package)
            logger.info(f"  ✓ {package}")
        except ImportError as e:
            logger.error(f"  ✗ {package}: {e}")
            return False
    
    logger.info("All imports verified!")
    return True


def create_directories():
    """Create required directory structure."""
    logger.info("Creating directory structure...")
    dirs = [
        'data/raw',
        'data/processed',
        'models',
        'logs/task1_eegnet',
        'logs/task2_biot',
        'analysis',
        'slides',
        'experiments/task1_eegnet',
        'experiments/task2_biot'
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"  ✓ {d}")


def test_cuda():
    """Check if CUDA is available."""
    logger.info("Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            logger.warning("  ✗ CUDA not available, will use CPU (training will be slow)")
            return False
    except Exception as e:
        logger.error(f"  Error checking CUDA: {e}")
        return False


def verify_data_loading():
    """Verify that data loading modules work."""
    logger.info("Verifying data loading module...")
    try:
        sys.path.insert(0, os.getcwd())
        from data.load_data import DataLoader
        logger.info("  ✓ DataLoader imported successfully")
        return True
    except Exception as e:
        logger.error(f"  ✗ Error importing DataLoader: {e}")
        return False


def verify_eegnet():
    """Verify EEGNet implementation."""
    logger.info("Verifying EEGNet implementation...")
    try:
        sys.path.insert(0, os.getcwd())
        import torch
        from models.eegnet import EEGNet
        
        # Test forward pass
        model = EEGNet(n_channels=22, n_classes=4, n_samples=1000)
        x = torch.randn(2, 22, 1000)
        out = model(x)
        
        assert out.shape == (2, 4), f"Expected output shape (2, 4), got {out.shape}"
        logger.info("  ✓ EEGNet forward pass works")
        return True
    except Exception as e:
        logger.error(f"  ✗ Error with EEGNet: {e}")
        return False


def main():
    """Run full setup."""
    logger.info("="*80)
    logger.info("EEG Foundation Model Assessment - SETUP")
    logger.info("="*80)
    
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Verifying imports", verify_imports),
        ("Creating directories", create_directories),
        ("Checking CUDA", test_cuda),
        ("Verifying data loading", verify_data_loading),
        ("Verifying EEGNet", verify_eegnet),
    ]
    
    for step_name, step_func in steps:
        logger.info(f"\n{step_name}...")
        try:
            step_func()
        except Exception as e:
            logger.error(f"Failed: {e}")
            return False
    
    logger.info("\n" + "="*80)
    logger.info("Setup complete! Ready to start training.")
    logger.info("Next step: python model/train_eegnet.py")
    logger.info("="*80)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
