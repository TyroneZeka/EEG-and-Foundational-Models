#!/usr/bin/env python
"""
Advanced EEGNet trainer with data augmentation and optimized hyperparameters.
Focus: Maximize accuracy over speed.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
import logging
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.eegnet import EEGNet
from data.load_data import DataLoader as EEGDataLoader
from data.preprocessing import EEGPreprocessor
from analysis.interpretability_analyzer import InterpretabilityAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGAugmentation:
    """Data augmentation for EEG signals."""
    
    @staticmethod
    def temporal_shift(X, max_shift=10):
        """Random temporal shift (circular)."""
        batch_size = X.shape[0]
        n_samples = X.shape[2]
        shifted = np.zeros_like(X)
        
        for i in range(batch_size):
            shift = np.random.randint(-max_shift, max_shift + 1)
            shifted[i] = np.roll(X[i], shift, axis=1)
        
        return shifted
    
    @staticmethod
    def gaussian_noise(X, noise_level=0.05):
        """Add Gaussian noise."""
        noise = np.random.randn(*X.shape) * noise_level * np.std(X)
        return X + noise
    
    @staticmethod
    def time_warp(X, warp_strength=0.1):
        """Simple time warping via interpolation."""
        batch_size, n_channels, n_samples = X.shape
        warped = np.zeros_like(X)
        
        for i in range(batch_size):
            # Create warping factor that varies per sample
            warp_factor = 1 + np.random.randn() * warp_strength
            new_indices = np.linspace(0, n_samples - 1, n_samples) / warp_factor
            new_indices = np.clip(new_indices, 0, n_samples - 1)
            
            for ch in range(n_channels):
                warped[i, ch] = np.interp(range(n_samples), new_indices, X[i, ch])
        
        return warped
    
    @staticmethod
    def mixup(X, y, alpha=0.2):
        """Mixup augmentation."""
        batch_size = X.shape[0]
        indices = np.random.permutation(batch_size)
        X_mixed = X.copy()
        y_mixed = y.copy()
        
        for i in range(batch_size // 2):
            lam = np.random.beta(alpha, alpha)
            idx1, idx2 = i, indices[i]
            X_mixed[idx1] = lam * X[idx1] + (1 - lam) * X[idx2]
            # For classification, keep original labels (could use soft labels)
        
        return X_mixed, y_mixed


class AugmentedEEGDataLoader:
    """DataLoader with on-the-fly augmentation."""
    
    def __init__(self, X, y, batch_size=32, shuffle=True, augment=False, augment_prob=0.3):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.augment_prob = augment_prob
        self.augmentor = EEGAugmentation()
        self.indices = np.arange(len(X))
        
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        for i in range(0, len(self.X), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices]
            
            # Apply augmentation
            if self.augment and np.random.random() < self.augment_prob:
                if np.random.random() < 0.33:
                    X_batch = self.augmentor.gaussian_noise(X_batch, noise_level=0.05)
                elif np.random.random() < 0.66:
                    X_batch = self.augmentor.temporal_shift(X_batch, max_shift=15)
                else:
                    X_batch = self.augmentor.time_warp(X_batch, warp_strength=0.1)
            
            yield torch.FloatTensor(X_batch), torch.LongTensor(y_batch)
    
    def __len__(self):
        return (len(self.X) + self.batch_size - 1) // self.batch_size


class AdvancedEEGNetTrainer:
    """Advanced trainer with augmentation and optimized hyperparameters."""
    
    def __init__(self, model, device='cpu', lr=0.0002, weight_decay=0.0002, epochs=500):
        self.model = model.to(device)
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Cosine annealing with warm restarts (T_mult must be integer)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )
        
        self.epochs = epochs
        self.best_val_acc = 0
        self.patience = 50
        self.patience_counter = 0
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for X, y in pbar:
            X, y = X.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().detach().numpy())
            all_labels.extend(y.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        
        return avg_loss, balanced_acc
    
    def evaluate(self, val_loader, phase='val'):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Evaluating ({phase})", leave=False)
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
    
    def train(self, X_train, y_train, X_val, y_val, X_test, y_test, log_dir='logs'):
        """Train with augmentation."""
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        
        logger.info(f"Starting training: {self.epochs} epochs with augmentation")
        logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        best_model_state = None
        
        for epoch in range(self.epochs):
            # Create augmented train loader
            train_loader = AugmentedEEGDataLoader(
                X_train, y_train, batch_size=32, shuffle=True, augment=True, augment_prob=0.5
            )
            
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validation (no augmentation)
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
                batch_size=32, shuffle=False
            )
            val_loss, val_acc, _, _, _ = self.evaluate(val_loader, phase='val')
            
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/balanced_accuracy', train_acc, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/balanced_accuracy', val_acc, epoch)
            writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            self.scheduler.step()
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                logger.info(f"Epoch {epoch+1}: ✓ New best val acc {val_acc:.4f}")
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % 50 == 0:
                logger.info(f"Epoch {epoch+1}: Train loss={train_loss:.4f}, Val acc={val_acc:.4f}, LR={self.optimizer.param_groups[0]['lr']:.6f}")
            
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Test evaluation
        self.model.load_state_dict(best_model_state)
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
            batch_size=32, shuffle=False
        )
        test_loss, test_acc, test_accuracy, test_preds, test_labels = self.evaluate(test_loader, phase='test')
        
        logger.info(f"✓ Final Test: Loss={test_loss:.4f}, Balanced Acc={test_acc:.4f}, Accuracy={test_accuracy:.4f}")
        
        results = {
            'test_loss': test_loss,
            'test_balanced_acc': test_acc,
            'test_accuracy': test_accuracy,
        }
        
        writer.close()
        return best_model_state, results


def train_dataset_optimized(dataset_name, log_dir='logs'):
    """Train dataset with optimized settings for maximum accuracy."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING ON {dataset_name} (OPTIMIZED FOR ACCURACY)")
    logger.info(f"{'='*80}\n")
    
    # Load data
    loader = EEGDataLoader()
    
    if dataset_name == 'BCI_IV_2a':
        data = loader.load_bci_2a()
        sampling_rate = 250
    elif dataset_name == 'BCI_IV_2b':
        data = loader.load_bci_2b()
        sampling_rate = 250
    elif dataset_name == 'PhysioNet_MI':
        data = loader.load_physionet_mi()
        sampling_rate = data['sampling_rate']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    X, y, metadata = data['X'], data['y'], data['metadata']
    
    # Preprocess (global filters)
    preprocessor = EEGPreprocessor(sampling_rate=sampling_rate)
    X = preprocessor.apply_average_reference(X)
    X = preprocessor.apply_bandpass_filter(X)
    
    # Convert labels to int
    if y.dtype.kind in ('U', 'S', 'O'):
        unique_labels = np.unique(y)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_map[label] for label in y], dtype=np.int64)
        logger.info(f"Label mapping: {label_map}")
    
    # LOSO CV
    subjects = np.unique(metadata['subject'])
    fold_results = []
    
    logger.info(f"Running LOSO CV with {len(subjects)} subjects (OPTIMIZED)...")
    
    for fold_idx, test_subject in enumerate(subjects):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_idx+1}/{len(subjects)}: Test Subject = {test_subject}")
        logger.info(f"{'='*60}")
        
        test_mask = metadata['subject'] == test_subject
        train_val_mask = ~test_mask
        
        X_train_val = X[train_val_mask]
        y_train_val = y[train_val_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        # Train/val split FIRST (80/20) before normalization
        n_train_val = len(X_train_val)
        n_train = int(0.8 * n_train_val)
        indices = np.arange(n_train_val)
        np.random.seed(42 + fold_idx)  # Different seed per fold
        np.random.shuffle(indices)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        X_train, y_train = X_train_val[train_indices], y_train_val[train_indices]
        X_val, y_val = X_train_val[val_indices], y_train_val[val_indices]
        
        # STRICT: Fit normalization ONLY on training data, apply to all three sets
        mean_train = np.mean(X_train, axis=(0, 2), keepdims=True)
        std_train = np.std(X_train, axis=(0, 2), keepdims=True)
        std_train[std_train == 0] = 1
        
        # Apply the SAME normalization to train, val, and test
        X_train = (X_train - mean_train) / std_train
        X_val = (X_val - mean_train) / std_train
        X_test = (X_test - mean_train) / std_train
        
        # Create model
        n_channels = X.shape[1]
        n_samples = X.shape[2]
        n_classes = len(np.unique(y))
        
        model = EEGNet(n_channels, n_classes, n_samples, F1=16, F2=32, D=4)
        
        # Train with augmentation
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = AdvancedEEGNetTrainer(
            model, 
            device=device, 
            lr=0.0002,
            weight_decay=0.0002,
            epochs=500
        )
        
        fold_log_dir = os.path.join(log_dir, dataset_name.replace('_', ''), f'fold_{fold_idx+1}')
        best_state, results = trainer.train(X_train, y_train, X_val, y_val, X_test, y_test, fold_log_dir)
        
        # Save best model state
        model_save_dir = f'experiments/task1_eegnet/{dataset_name}'
        os.makedirs(model_save_dir, exist_ok=True)
        model_path = os.path.join(model_save_dir, f'fold_{fold_idx+1}_best_model.pth')
        torch.save(best_state, model_path)
        logger.info(f"Saved model to {model_path}")
        
        results['fold'] = fold_idx + 1
        results['subject'] = test_subject
        fold_results.append(results)
        
        # Run interpretability analysis on test set
        try:
            logger.info(f"\nRunning Figure 3 Analysis for Fold {fold_idx+1}...")
            model_for_analysis = EEGNet(n_channels, n_classes, n_samples, F1=16, F2=32, D=4)
            model_for_analysis.load_state_dict(best_state)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_for_analysis = model_for_analysis.to(device)
            
            analyzer = InterpretabilityAnalyzer(
                model=model_for_analysis,
                dataset_name=dataset_name,
                n_channels=n_channels,
                n_classes=n_classes,
                sampling_rate=250
            )
            
            analysis_output_dir = os.path.join('analysis/figure3_results', dataset_name)
            analyzer.analyze_fold(X_test, y_test, analysis_output_dir, fold_idx=fold_idx+1)
        except Exception as e:
            logger.warning(f"Figure 3 analysis failed for fold {fold_idx+1}: {e}")
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"{dataset_name} FINAL RESULTS")
    logger.info(f"{'='*80}")
    test_accs = [r['test_balanced_acc'] for r in fold_results]
    logger.info(f"Mean Balanced Accuracy: {np.mean(test_accs):.4f} +/- {np.std(test_accs):.4f}")
    logger.info(f"Min/Max: {np.min(test_accs):.4f} / {np.max(test_accs):.4f}")
    logger.info(f"Per-fold accuracies: {[f'{acc:.4f}' for acc in test_accs]}")
    logger.info(f"{'='*80}\n")
    
    return fold_results


if __name__ == "__main__":
    # Train all datasets with optimization focus
    results_2a = train_dataset_optimized('BCI_IV_2a', log_dir='logs/task1_eegnet_optimized')
    results_2b = train_dataset_optimized('BCI_IV_2b', log_dir='logs/task1_eegnet_optimized')
    results_physionet = train_dataset_optimized('PhysioNet_MI', log_dir='logs/task1_eegnet_optimized')
    
    logger.info("\n" + "="*80)
    logger.info("ALL DATASETS TRAINING COMPLETE")
    logger.info("="*80)
