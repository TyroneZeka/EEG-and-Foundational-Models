#!/usr/bin/env python
"""
Optimized training script for BCI_IV_2b and PhysioNet_MI datasets.
Uses better hyperparameters for improved generalization.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.eegnet import EEGNet
from data.load_data import DataLoader as EEGDataLoader
from data.preprocessing import EEGPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedEEGNetTrainer:
    """Improved trainer with better hyperparameters."""
    
    def __init__(self, model, device='cpu', lr=0.001, batch_size=32, epochs=200):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
        # Learning rate scheduler: cosine annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.epochs = epochs
        self.batch_size = batch_size
    
    def train_epoch(self, train_loader, epoch):
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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().detach().numpy())
            all_labels.extend(y.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        
        return avg_loss, balanced_acc
    
    def evaluate(self, val_loader, phase='val'):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Evaluating ({phase})")
            for X, y in pbar:
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
        
        return avg_loss, balanced_acc, accuracy, all_preds, all_labels
    
    def train(self, train_loader, val_loader, test_loader=None, log_dir='logs'):
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        
        best_val_acc = 0
        best_model_state = None
        patience = 20
        patience_counter = 0
        
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc, _, _, _ = self.evaluate(val_loader, phase='val')
            
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/balanced_accuracy', train_acc, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/balanced_accuracy', val_acc, epoch)
            
            self.scheduler.step()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                logger.info(f"Epoch {epoch+1}: New best val acc {val_acc:.4f}")
            else:
                patience_counter += 1
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}: Train loss={train_loss:.4f}, Val acc={val_acc:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Test evaluation
        if test_loader:
            self.model.load_state_dict(best_model_state)
            test_loss, test_acc, test_accuracy, test_preds, test_labels = self.evaluate(test_loader, phase='test')
            logger.info(f"\n✓ Test: Loss={test_loss:.4f}, Balanced Acc={test_acc:.4f}, Accuracy={test_accuracy:.4f}")
            results = {
                'test_loss': test_loss,
                'test_balanced_acc': test_acc,
                'test_accuracy': test_accuracy,
            }
        else:
            results = {}
        
        writer.close()
        return best_model_state, results


def train_dataset(dataset_name, log_dir='logs'):
    """Train on a dataset with LOSO CV."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING ON {dataset_name}")
    logger.info(f"{'='*80}\n")
    
    # Load data
    loader = EEGDataLoader()
    
    if dataset_name == 'BCI_IV_2b':
        data = loader.load_bci_2b()
    elif dataset_name == 'PhysioNet_MI':
        data = loader.load_physionet_mi()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    X, y, metadata = data['X'], data['y'], data['metadata']
    
    # Preprocess
    preprocessor = EEGPreprocessor(sampling_rate=data['sampling_rate'])
    X = preprocessor.apply_average_reference(X)
    X = preprocessor.apply_bandpass_filter(X)
    
    # Convert labels to int
    if y.dtype.kind in ('U', 'S', 'O'):
        unique_labels = np.unique(y)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_map[label] for label in y], dtype=np.int64)
    
    # LOSO CV
    subjects = np.unique(metadata['subject'])
    fold_results = []
    
    logger.info(f"Running LOSO CV with {len(subjects)} subjects...")
    
    for fold_idx, test_subject in enumerate(subjects):
        logger.info(f"\nFold {fold_idx+1}/{len(subjects)}: Test Subject = {test_subject}")
        
        test_mask = metadata['subject'] == test_subject
        train_val_mask = ~test_mask
        
        X_train_val = X[train_val_mask]
        y_train_val = y[train_val_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        # Per-fold z-score normalization
        mean_val = np.mean(X_train_val, axis=(0, 2), keepdims=True)
        std_val = np.std(X_train_val, axis=(0, 2), keepdims=True)
        std_val[std_val == 0] = 1
        
        X_train_val = (X_train_val - mean_val) / std_val
        X_test = (X_test - mean_val) / std_val
        
        # Train/val split
        n_train_val = len(X_train_val)
        n_train = int(0.8 * n_train_val)
        indices = np.arange(n_train_val)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        X_train, y_train = X_train_val[train_indices], y_train_val[train_indices]
        X_val, y_val = X_train_val[val_indices], y_train_val[val_indices]
        
        # Create dataloaders
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
        
        # Create model
        n_channels = X.shape[1]
        n_samples = X.shape[2]
        n_classes = len(np.unique(y))
        
        model = EEGNet(n_channels, n_classes, n_samples)
        
        # Train
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = OptimizedEEGNetTrainer(model, device=device, epochs=200)
        fold_log_dir = os.path.join(log_dir, dataset_name.replace('_', ''), f'fold_{fold_idx+1}')
        best_state, results = trainer.train(train_loader, val_loader, test_loader, fold_log_dir)
        
        results['fold'] = fold_idx + 1
        results['subject'] = test_subject
        fold_results.append(results)
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"{dataset_name} SUMMARY")
    logger.info(f"{'='*80}")
    test_accs = [r['test_balanced_acc'] for r in fold_results]
    logger.info(f"Mean Balanced Accuracy: {np.mean(test_accs):.4f} +/- {np.std(test_accs):.4f}")
    logger.info(f"Per-fold accuracies: {[f'{acc:.4f}' for acc in test_accs]}")
    
    return fold_results


if __name__ == "__main__":
    # Train on both datasets
    results_2b = train_dataset('BCI_IV_2b', log_dir='logs/task1_eegnet')
    results_physionet = train_dataset('PhysioNet_MI', log_dir='logs/task1_eegnet')
    
    logger.info("\n" + "="*80)
    logger.info("ALL DATASETS COMPLETE")
    logger.info("="*80)
