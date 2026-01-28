"""
Training script for EEGNet with Leave-One-Subject-Out (LOSO) cross-validation.
Logs all metrics to TensorBoard.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
import logging
from tqdm import tqdm
import sys

# Add project to path (portable for any server)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.eegnet import EEGNet
from data.load_data import DataLoader as EEGDataLoader
from data.preprocessing import preprocess_dataset, EEGPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGNetTrainer:
    """Trainer class for EEGNet with LOSO CV."""
    
    def __init__(self,
                 model,
                 device='cpu',
                 lr=0.001,
                 batch_size=32,
                 epochs=100,
                 random_seed=42):
        """
        Args:
            model: EEGNet model
            device: 'cpu' or 'cuda'
            lr: Learning rate
            batch_size: Batch size
            epochs: Number of training epochs
            random_seed: For reproducibility
        """
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_seed = random_seed
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    def train_epoch(self, train_loader, writer=None, epoch=0):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, (X, y) in enumerate(tqdm(train_loader, desc="Training")):
            X, y = X.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        
        if writer:
            writer.add_scalar('train/loss', avg_loss, epoch)
            writer.add_scalar('train/balanced_accuracy', balanced_acc, epoch)
        
        return avg_loss, balanced_acc
    
    def evaluate(self, val_loader, writer=None, epoch=0, phase='val'):
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X, y in tqdm(val_loader, desc=f"Evaluating ({phase})"):
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
        
        if writer:
            writer.add_scalar(f'{phase}/loss', avg_loss, epoch)
            writer.add_scalar(f'{phase}/balanced_accuracy', balanced_acc, epoch)
            writer.add_scalar(f'{phase}/accuracy', accuracy, epoch)
        
        return avg_loss, balanced_acc, accuracy, all_preds, all_labels
    
    def train(self, train_loader, val_loader, test_loader=None, log_dir='logs/task1_eegnet'):
        """
        Train model with validation and testing.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            test_loader: Test DataLoader (optional)
            log_dir: TensorBoard log directory
        
        Returns:
            best_model_state, results
        """
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        
        best_val_acc = 0
        best_model_state = None
        results = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        patience = 20
        patience_counter = 0
        
        logger.info(f"Starting training for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(train_loader, writer, epoch)
            val_loss, val_acc, _, _, _ = self.evaluate(val_loader, writer, epoch, phase='val')
            
            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)
            results['val_loss'].append(val_loss)
            results['val_acc'].append(val_acc)
            
            self.scheduler.step()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                logger.info(f"Epoch {epoch+1}: New best validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Test evaluation
        if test_loader:
            self.model.load_state_dict(best_model_state)
            test_loss, test_acc, test_accuracy, test_preds, test_labels = self.evaluate(
                test_loader, writer, 0, phase='test'
            )
            results['test_loss'] = test_loss
            results['test_balanced_acc'] = test_acc
            results['test_accuracy'] = test_accuracy
            results['test_preds'] = test_preds
            results['test_labels'] = test_labels
            
            logger.info(f"Test: Loss={test_loss:.4f}, Balanced Acc={test_acc:.4f}, Accuracy={test_accuracy:.4f}")
        
        writer.close()
        return best_model_state, results


def loso_cross_validation(X, y, metadata, dataset_name='BCI_IV_2a', log_dir='logs/task1_eegnet'):
    """
    Leave-One-Subject-Out cross-validation.
    
    Args:
        X: Data (n_samples, n_channels, n_times) - PREPROCESSED
        y: Labels (n_samples,)
        metadata: Metadata with 'subject' column
        dataset_name: Name of dataset
        log_dir: TensorBoard log directory
    
    Returns:
        fold_results: List of results per fold
    """
    subjects = np.unique(metadata['subject'])
    fold_results = []
    
    logger.info(f"Starting LOSO cross-validation on {dataset_name}...")
    logger.info(f"Number of subjects: {len(subjects)}")
    
    # Preprocessor for within-fold normalization
    preprocessor = EEGPreprocessor(sampling_rate=250)
    
    for fold_idx, test_subject in enumerate(subjects):
        logger.info(f"\n{'='*80}")
        logger.info(f"Fold {fold_idx+1}/{len(subjects)}: Test Subject = {test_subject}")
        logger.info(f"{'='*80}")
        
        # Split into train+val and test
        test_mask = metadata['subject'] == test_subject
        train_val_mask = ~test_mask
        
        X_train_val = X[train_val_mask]
        y_train_val = y[train_val_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        # Apply z-score normalization WITHIN fold (fit on train+val only)
        # This prevents test data leakage into normalization statistics
        from scipy import signal
        
        # Apply bandpass filter to this fold (already done globally, but re-apply for safety)
        # and z-score normalize using ONLY train+val statistics
        mean_val = np.mean(X_train_val, axis=(0, 2), keepdims=True)
        std_val = np.std(X_train_val, axis=(0, 2), keepdims=True)
        std_val[std_val == 0] = 1
        
        X_train_val = (X_train_val - mean_val) / std_val
        X_test = (X_test - mean_val) / std_val
        
        logger.info(f"Applied within-fold z-score normalization (fit on train+val only)")
        
        # Further split train+val into train and val (80-20)
        n_train_val = len(X_train_val)
        n_train = int(0.8 * n_train_val)
        indices = np.arange(n_train_val)
        np.random.shuffle(indices)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        X_train, y_train = X_train_val[train_indices], y_train_val[train_indices]
        X_val, y_val = X_train_val[val_indices], y_train_val[val_indices]
        
        # Convert to torch tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.LongTensor(y_test)
        
        # Create dataloaders
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32)
        test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32)
        
        # Create model
        n_channels = X.shape[1]
        n_samples = X.shape[2]
        n_classes = len(np.unique(y))
        
        model = EEGNet(n_channels, n_classes, n_samples)
        
        # Train
        trainer = EEGNetTrainer(model, device='cuda' if torch.cuda.is_available() else 'cpu')
        fold_log_dir = os.path.join(log_dir, dataset_name, f'fold_{fold_idx+1}')
        best_state, results = trainer.train(train_loader, val_loader, test_loader, fold_log_dir)
        
        results['fold'] = fold_idx + 1
        results['test_subject'] = test_subject
        fold_results.append(results)
    
    return fold_results


if __name__ == "__main__":
    # Load data
    logger.info("Loading datasets...")
    loader = EEGDataLoader()
    
    # Load BCI_IV_2a
    data_2a = loader.load_bci_2a()
    
    # Apply initial preprocessing (average reference, bandpass) 
    # Z-score normalization will be done per-fold to prevent test leakage
    preprocessor = EEGPreprocessor(sampling_rate=250)
    X_processed = preprocessor.apply_average_reference(data_2a['X'])
    X_processed = preprocessor.apply_bandpass_filter(X_processed)
    data_2a['X'] = X_processed
    # Convert labels to int (if they're strings from MOABB)
    if data_2a['y'].dtype.kind in ('U', 'S', 'O'):
        unique_labels = np.unique(data_2a['y'])
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        data_2a['y'] = np.array([label_map[label] for label in data_2a['y']], dtype=np.int64)
    
    # Run LOSO CV
    fold_results = loso_cross_validation(
        data_2a['X'],
        data_2a['y'],
        data_2a['metadata'],
        dataset_name='BCI_IV_2a',
        log_dir='logs/task1_eegnet'
    )
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("LOSO CROSS-VALIDATION SUMMARY")
    logger.info("="*80)
    test_accs = [r['test_balanced_acc'] for r in fold_results]
    logger.info(f"Mean Balanced Accuracy: {np.mean(test_accs):.4f} +/- {np.std(test_accs):.4f}")
    logger.info(f"Per-fold accuracies: {[f'{acc:.4f}' for acc in test_accs]}")
