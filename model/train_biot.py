import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from torch.utils.tensorboard import SummaryWriter # Import SummaryWriter
from pathlib import Path
import logging

from tqdm import tqdm

# --- Setup Paths and Logging ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.biot import BIOTClassifier
from data.load_data import DataLoader as EEGDataLoader
from data.preprocessing import EEGPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main BIOT Trainer Class ---

class BIOT_Trainer:
    def __init__(self, model_config, lr=0.0001, epochs=100, patience=15, device='cuda'):
        self.model_config = model_config
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    # In your train.py file, inside the BIOT_Trainer class

    def _init_model(self, mode, pretrained_path=None):
        """Initializes the BIOTClassifier model based on the mode."""
        model = BIOTClassifier(**self.model_config).to(self.device)
        
        if mode == 'fine_tune':
            if pretrained_path is None or not os.path.exists(pretrained_path):
                raise FileNotFoundError(f"Pretrained model path not found: {pretrained_path}")
            logging.info(f"Loading pretrained weights for fine-tuning from: {pretrained_path}")
            pretrained_dict = torch.load(pretrained_path, map_location=self.device)
            model_dict = model.state_dict()
            pretrained_dict_adapted = {f"biot.{k}": v for k, v in pretrained_dict.items() if f"biot.{k}" in model_dict}
            keys_to_exclude = ["biot.channel_tokens.weight", "biot.index"]
            pretrained_dict_adapted = {k: v for k, v in pretrained_dict_adapted.items() if k not in keys_to_exclude}

            model_dict.update(pretrained_dict_adapted)
            model.load_state_dict(model_dict, strict=False)

        elif mode == 'from_scratch':
            logging.info("Initializing BIOTClassifier with random weights (from scratch).")

        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        return model

    def run(self, X, y, mode, log_dir, save_path, pretrained_path=None):
        """Runs the full training and evaluation pipeline with TensorBoard logging."""
        
        # --- TENSORBOARD SETUP ---
        writer = SummaryWriter(log_dir)
        logging.info(f"TensorBoard logs will be saved to: {log_dir}")
        # --- END OF SETUP ---
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=32, shuffle=False)
        test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=32, shuffle=False)
        
        model = self._init_model(mode, pretrained_path)
        
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(self.epochs):
            # --- Training Epoch ---
            model.train()
            train_losses = []
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
            for batch_X, batch_y in pbar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            writer.add_scalar('Loss/train', avg_train_loss, epoch)

            model.eval()
            val_losses, all_preds, all_labels = [], [], []
            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Validation]", leave=False)
                for batch_X, batch_y in pbar:
                    batch_X = batch_X.to(self.device)
                    outputs = model(batch_X)
                    loss = self.criterion(outputs, batch_y.to(self.device))
                    val_losses.append(loss.item())
                    all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
                    all_labels.append(batch_y.cpu().numpy())
            
            avg_val_loss = np.mean(val_losses)
            val_acc = balanced_accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))
            
            writer.add_scalar('Loss/validation', avg_val_loss, epoch)
            writer.add_scalar('Accuracy/validation', val_acc, epoch)
            
            if (epoch + 1) % 10 == 0: logging.info(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f} - Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save the state dictionary of the best model
                best_model_state = model.state_dict()
                patience_counter = 0
                logging.info(f"✓ Epoch {epoch+1}/{self.epochs} - New best val acc: {val_acc:.4f}. Model state saved.")
            else:
                patience_counter += 1
            if patience_counter >= self.patience: logging.info(f"Early stopping at epoch {epoch+1}."); break
            self.scheduler.step()
            
        if best_model_state is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(best_model_state, save_path)
            logging.info(f"Best model saved to {save_path}")
            # Load the best model for final testing
            model.load_state_dict(best_model_state)

        model.eval()
        test_losses, all_preds, all_labels = [], [], []
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Epoch {epoch+1} [Test]", leave=False)
            for batch_X, batch_y in pbar:
                batch_X = batch_X.to(self.device)
                outputs = model(batch_X)
                loss = self.criterion(outputs, batch_y.to(self.device))
                test_losses.append(loss.item())
                all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())
        
        avg_test_loss = np.mean(test_losses)
        test_acc = balanced_accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))
        
        writer.add_scalar('Loss/test', avg_test_loss, 0)
        writer.add_scalar('Accuracy/test', test_acc, 0)
        
        logging.info(f"--- FINAL RESULT (Mode: {mode}) --- Test Balanced Accuracy: {test_acc:.4f} ---")
        writer.close()
        return test_acc

# In train_biot.py

def main():
    """
    Main function to configure and run the BIOT training experiment for multiple datasets.
    """
    # --- MASTER CONFIGURATION ---
    PRETRAINED_PATH = 'pretrained_biot/EEG-six-datasets-18-channels.ckpt'
    BASE_LOG_DIR = 'logs/task2_biot'
    
    DATASETS_TO_RUN = ['BCI_IV_2a', 'BNCI2015_001'] # Add the second dataset here
    # --- END OF CONFIGURATION ---

    data_loader = EEGDataLoader()
    
    for dataset_name in DATASETS_TO_RUN:
        logging.info(f"\n{'='*80}\nSTARTING EXPERIMENTS FOR DATASET: {dataset_name}\n{'='*80}")
        
        # Load and preprocess data dynamically
        preprocessor = EEGPreprocessor(sampling_rate=250)
        # loader_func = getattr(data_loader, f"load_{dataset_name.lower()}")
        if dataset_name == 'BNCI2015_001':
            loader_func = data_loader.load_bnci2015_001 
        else:
            loader_func = data_loader.load_bci_2a  
        data = loader_func()
        X, y = preprocessor.preprocess(data['X'], data['y'])

        # Define the BIOT model configuration dynamically
        model_config = {
            'n_channels': X.shape[1],
            'n_classes': len(np.unique(y)),
            'emb_size': 256,
            'heads': 8,
            'depth': 4,
            'n_fft': 200,
            'hop_length': 20
        }

        # Instantiate the trainer and run both modes
        trainer = BIOT_Trainer(model_config)
        
        # Run 1: From Scratch
        logging.info(f"\n--- Running Experiment: {dataset_name} - FROM SCRATCH ---")
        trainer.run(X, y, mode='from_scratch',
                    log_dir=f"{BASE_LOG_DIR}/{dataset_name}/from_scratch",
                    save_path=f"models/biot/{dataset_name}_from_scratch.pth")
        
        # Run 2: Fine-Tuning
        logging.info(f"\n--- Running Experiment: {dataset_name} - FINE-TUNING ---")
        trainer.run(X, y, mode='fine_tune',
                    pretrained_path=PRETRAINED_PATH,
                    log_dir=f"{BASE_LOG_DIR}/{dataset_name}/fine_tune",
                    save_path=f"models/biot/{dataset_name}_fine_tune.pth")

if __name__ == "__main__":
    main()

