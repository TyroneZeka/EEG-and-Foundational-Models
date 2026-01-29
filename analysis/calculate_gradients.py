import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# --- Setup Paths and Logging ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.eegnet import EEGNet
from data.load_data import DataLoader as EEGDataLoader
from data.preprocessing import EEGPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_gradient_distribution(model, data_loader, device):
    """
    Performs one forward/backward pass and returns all gradients as a single flat array.
    """
    model.train() # Set to train mode to ensure gradients are computed
    
    # Get a single batch of data
    X_batch, y_batch = next(iter(data_loader))
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

    # Forward pass
    outputs = model(X_batch)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, y_batch)
    
    # Backward pass to calculate gradients
    loss.backward()
    
    # Collect all gradients
    all_grads = []
    for param in model.parameters():
        if param.grad is not None:
            all_grads.append(param.grad.cpu().numpy().flatten())
            
    # VERY IMPORTANT: Clear the gradients after collecting them
    model.zero_grad()
    
    return np.concatenate(all_grads)

def plot_gradient_histograms(grad_begin, grad_end, output_path):
    """Plots two gradient distributions side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot "Beginning" gradients
    ax1.hist(grad_begin, bins=100, color='royalblue', alpha=0.8)
    ax1.set_title('Beginning of Training (Random Weights)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Gradient Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot "End" gradients
    ax2.hist(grad_end, bins=100, color='seagreen', alpha=0.8)
    ax2.set_title('End of Training (Converged Weights)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Gradient Value')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    fig.suptitle('Comparison of Gradient Distributions', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"✓ Saved gradient comparison plot to {output_path}")

# --- Main Execution Logic ---
def main():
    """
    Main function to re-calculate and visualize gradient distributions.
    """
    # --- USER CONFIGURATION ---
    DATASET_NAME = 'BCI_IV_2a'
    FOLD_NUM = 1
    
    # Model Hyperparameters (MUST match the trained model)
    MODEL_HYPERPARAMS = {'F1': 16, 'D': 4, 'F2': 32}
    
    OUTPUT_DIR = f'analysis/gradient_analysis/{DATASET_NAME}/fold_{FOLD_NUM}'
    # --- END OF CONFIGURATION ---

    # 1. Load a single batch of data
    data_loader_manager = EEGDataLoader()
    preprocessor = EEGPreprocessor(sampling_rate=250)
    loader_func = getattr(data_loader_manager, 'load_bci_2a')
    data = loader_func()
    X, y = preprocessor.preprocess(data['X'], data['y'])
    
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    n_channels = X.shape[1]
    n_samples = X.shape[2]
    n_classes = len(np.unique(y))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. Get gradients for "Beginning" state
    logging.info("Calculating gradients for 'Beginning' state (random weights)...")
    model_begin = EEGNet(
        n_channels=n_channels, n_classes=n_classes, n_samples=n_samples,
        **MODEL_HYPERPARAMS
    ).to(device)
    grads_begin = get_gradient_distribution(model_begin, data_loader, device)

    # 3. Get gradients for "End" state
    logging.info("Calculating gradients for 'End' state (trained weights)...")
    model_path = f'experiments/task1_eegnet/{DATASET_NAME}/fold_{FOLD_NUM}_best_model.pth'
    if not os.path.exists(model_path):
        logging.error(f"Model not found at {model_path}. Cannot calculate 'End' state gradients.")
        return
        
    model_end = EEGNet(
        n_channels=n_channels, n_classes=n_classes, n_samples=n_samples,
        **MODEL_HYPERPARAMS
    ).to(device)
    model_end.load_state_dict(torch.load(model_path, map_location=device))
    grads_end = get_gradient_distribution(model_end, data_loader, device)

    # 4. Plot the comparison
    plot_gradient_histograms(grads_begin, grads_end, output_path=f"{OUTPUT_DIR}/gradient_distribution_comparison.png")

if __name__ == "__main__":
    main()
