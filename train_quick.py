#!/usr/bin/env python
"""
Quick training script to test pipeline on BCI_IV_2a.
This script will download the dataset and train EEGNet on a single fold.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import logging
from tqdm import tqdm

sys.path.insert(0, '.')
from models.eegnet import EEGNet
from data.preprocessing import EEGPreprocessor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def download_bci2a_data():
    """Download BCI_IV_2a via MOABB."""
    logger.info("Downloading BCI_IV_2a dataset...")
    try:
        from moabb.datasets import BNCI2014001
        from moabb.paradigms import MotorImagery
        
        dataset = BNCI2014001()
        paradigm = MotorImagery(n_classes=4)
        X, y, metadata = paradigm.get_data(dataset=dataset)
        
        logger.info(f"✓ BCI_IV_2a loaded: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"  Subjects: {len(np.unique(metadata['subject']))}")
        logger.info(f"  Classes: {np.unique(y)}")
        
        return X, y, metadata
    except Exception as e:
        logger.error(f"✗ Failed to download BCI_IV_2a: {e}")
        # Return dummy data for testing
        logger.warning("Using dummy data for testing...")
        X = np.random.randn(100, 22, 1000)
        y = np.random.randint(0, 4, 100)
        metadata = {'subject': np.array([i // 25 for i in range(100)])}
        return X, y, metadata


def preprocess_data(X, y):
    """Preprocess EEG data."""
    logger.info("Preprocessing data...")
    preprocessor = EEGPreprocessor(sampling_rate=250)
    X_proc, y_proc = preprocessor.preprocess(X, y)
    logger.info(f"✓ Preprocessing complete: {X.shape} -> {X_proc.shape}")
    return X_proc, y_proc


class QuickTrainer:
    """Simple trainer for quick testing."""
    
    def __init__(self, model, device='cpu', lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
    
    def train_epoch(self, train_loader, epoch):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for X, y in pbar:
            X, y = X.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        
        return avg_loss, balanced_acc
    
    def evaluate(self, val_loader, phase='val'):
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X, y in tqdm(val_loader, desc=f"{phase.upper()}"):
                X, y = X.to(self.device), y.to(self.device)
                
                logits = self.model(X)
                loss = self.criterion(logits, y)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, balanced_acc, accuracy
    
    def train(self, train_loader, val_loader, test_loader, epochs=30, log_dir='logs/quick_test'):
        """Train model."""
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        
        best_val_acc = 0
        
        logger.info(f"Training for {epochs} epochs...")
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc, _ = self.evaluate(val_loader, phase='val')
            
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/balanced_accuracy', train_acc, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/balanced_accuracy', val_acc, epoch)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logger.info(f"✓ Epoch {epoch+1}: New best val acc {val_acc:.4f}")
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}: Train loss={train_loss:.4f}, Val acc={val_acc:.4f}")
        
        # Final test evaluation
        test_loss, test_acc, test_accuracy = self.evaluate(test_loader, phase='test')
        logger.info(f"\n✓ Test: Loss={test_loss:.4f}, Balanced Acc={test_acc:.4f}, Accuracy={test_accuracy:.4f}")
        
        writer.close()
        return test_acc


def main():
    """Run training pipeline."""
    logger.info("="*80)
    logger.info("EEGNet Quick Training Pipeline")
    logger.info("="*80 + "\n")
    
    # 1. Download and load data
    X, y, metadata = download_bci2a_data()
    
    # 2. Preprocess
    X, y = preprocess_data(X, y)
    
    # 3. Create train/val/test split (simple: 60/20/20)
    n_samples = len(X)
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    logger.info(f"Data split: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}\n")
    
    # 4. Create dataloaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=32
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=32
    )
    
    # 5. Create model
    n_classes = len(np.unique(y))
    n_channels = X.shape[1]
    n_samples_eeg = X.shape[2]
    
    logger.info(f"Creating EEGNet model: {n_channels} channels, {n_classes} classes, {n_samples_eeg} samples\n")
    model = EEGNet(n_channels, n_classes, n_samples_eeg)
    
    # 6. Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}\n")
    
    trainer = QuickTrainer(model, device=device, lr=0.001)
    test_acc = trainer.train(train_loader, val_loader, test_loader, epochs=30, log_dir='logs/task1_eegnet/quick_test')
    
    logger.info("\n" + "="*80)
    logger.info(f"Training complete! Test balanced accuracy: {test_acc:.4f}")
    logger.info("TensorBoard logs saved to: logs/task1_eegnet/quick_test")
    logger.info("View with: tensorboard --logdir logs/task1_eegnet/quick_test")
    logger.info("="*80)


if __name__ == "__main__":
    main()
