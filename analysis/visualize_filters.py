import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mne
from pathlib import Path
import logging

# --- Setup Paths and Logging ---
# Add the project root to the path to allow imports from other directories
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.eegnet import EEGNet # Assumes your EEGNet is in models/eegnet.py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Visualization Functions ---

# In visualize_filters.py

def visualize_spatial_filters(model, n_channels, sfreq, output_path):
    """
    Generates and saves topographic maps for the individual spatial filters
    learned by the model's depthwise convolutional layer.
    """
    logging.info("Visualizing spatial filters from the depthwise layer...")
    
    # 1. Extract weights
    try:
        spatial_layer = model.depthwise
        spatial_weights = spatial_layer.weight.data.cpu().numpy().squeeze()
    except AttributeError:
        logging.error("Could not find a 'depthwise' layer in the model. Cannot visualize spatial filters.")
        return

    # --- THIS IS THE FIX ---
    # 2. Use the EXACT channel names for the BCI_IV_2a dataset (22 channels)
    # This ensures the data from the model aligns perfectly with the electrode locations.
    dataset_specific_ch_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
        'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
    ]

    # Ensure the number of channels matches the model
    if n_channels != len(dataset_specific_ch_names):
        logging.error(f"Model channel count ({n_channels}) does not match the provided list ({len(dataset_specific_ch_names)}).")
        # As a fallback, use generic names, but this will likely result in the same error.
        montage = mne.channels.make_standard_montage('standard_1020')
        ch_names = montage.ch_names[:n_channels]
    else:
        ch_names = dataset_specific_ch_names

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    info.set_montage('standard_1020', on_missing='warn')
    # --- END OF FIX ---

    # 3. Determine number of filters to plot
    try:
        n_filters_to_plot = model.conv1.out_channels
    except AttributeError:
        logging.error("Could not find a 'conv1' layer to determine F1. Cannot proceed.")
        return

    n_cols = 4
    n_rows = int(np.ceil(n_filters_to_plot / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows + 1),
                           gridspec_kw={'hspace': 0.4, 'wspace': 0.1})
    
    vmax = np.max(np.abs(spatial_weights[:n_filters_to_plot]))
    vmin = -vmax

    for i, ax in enumerate(axes.flat):
        if i < n_filters_to_plot:
            mne.viz.plot_topomap(spatial_weights[i], info, axes=ax, show=False,
                                 cmap='RdBu_r', vlim=(vmin, vmax))
            ax.set_title(f'Filter {i+1}')
        else:
            ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label='Filter Weight Intensity')

    fig.suptitle('Spatial Filters (from Depthwise Layer)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # 4. Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"✓ Saved spatial filter topomaps to {output_path}")


def visualize_temporal_filters(model, sfreq, output_path):
    """
    Generates and saves a plot of the frequency responses for all
    individual temporal filters learned by the model's first convolutional layer.
    """
    logging.info("Visualizing temporal filter frequency responses...")
    
    # 1. Extract weights from the first convolutional layer
    try:
        temporal_layer = model.conv1
        # Shape: (F1, 1, 1, kernel_length) -> squeeze to (F1, kernel_length)
        temporal_kernels = temporal_layer.weight.data.cpu().numpy().squeeze()
    except AttributeError:
        logging.error("Could not find a 'conv1' layer in the model. Cannot visualize temporal filters.")
        return

    kernel_length = temporal_kernels.shape[1]
    fig, ax = plt.subplots(figsize=(12, 7))

    # 2. Compute and plot the frequency response for each individual kernel
    for kernel in temporal_kernels:
        freqs = np.fft.rfftfreq(kernel_length, 1 / sfreq)
        fft_vals = np.abs(np.fft.rfft(kernel))
        ax.plot(freqs, fft_vals, color='steelblue', alpha=0.5, linewidth=1.5)

    # 3. Style the plot for clarity
    ax.set_title(f'Spectral Response of all {len(temporal_kernels)} Temporal Filters', fontsize=16, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Response Magnitude')
    ax.set_xlim(1, 40) # Focus on the most relevant EEG frequencies (1Hz to 40Hz)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axvspan(8, 12, color='red', alpha=0.1, label='Alpha Band (8-12 Hz)')
    ax.axvspan(13, 30, color='blue', alpha=0.1, label='Beta Band (13-30 Hz)')
    ax.legend()
    
    # 4. Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"✓ Saved spectral filter plot to {output_path}")

# --- Main Execution ---

def main():
    """
    Main function to run the visualization analysis.
    Set the parameters below before running.
    """
    # --- USER CONFIGURATION ---
    DATASET_NAME = 'BCI_IV_2a'
    FOLD_IDX = 1  # Which fold to visualize (1-9 for BCI_IV_2a, 1-9 for BCI_IV_2b, 1-5 for PhysioNet)
    
    # Determine dataset parameters
    dataset_params = {
        'BCI_IV_2a': {'n_channels': 22, 'n_classes': 4, 'n_samples': 1000, 'srate': 250},
        'BCI_IV_2b': {'n_channels': 22, 'n_classes': 2, 'n_samples': 1000, 'srate': 250},
        'PhysioNet_MI': {'n_channels': 64, 'n_classes': 3, 'n_samples': 1600, 'srate': 160},
    }
    
    if DATASET_NAME not in dataset_params:
        logging.error(f"Unknown dataset: {DATASET_NAME}")
        return
    
    params = dataset_params[DATASET_NAME]
    N_CHANNELS = params['n_channels']
    N_CLASSES = params['n_classes']
    N_SAMPLES = params['n_samples']
    SAMPLING_RATE = params['srate']
    
    # Model path (where training saves models)
    MODEL_PATH = f'experiments/task1_eegnet/{DATASET_NAME}/fold_{FOLD_IDX}_best_model.pth'
    OUTPUT_DIR = f'analysis/figures/{DATASET_NAME}/fold_{FOLD_IDX}'
    # --- END OF CONFIGURATION ---

    # 1. Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        logging.error(f"FATAL: Model file not found at '{MODEL_PATH}'")
        logging.info(f"Expected location: {os.path.abspath(MODEL_PATH)}")
        logging.info("Make sure you've run: python model/train_eegnet_optimized.py")
        return

    # 2. Initialize and load the trained model
    logging.info(f"Loading trained model from {MODEL_PATH}...")
    model = EEGNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, n_samples=N_SAMPLES, F1=16, F2=32, D=4)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    logging.info("✓ Model loaded successfully")

    # 3. Generate and save the visualizations
    visualize_spatial_filters(model, N_CHANNELS, SAMPLING_RATE, output_path=f"{OUTPUT_DIR}/spatial_filters.png")
    visualize_temporal_filters(model, SAMPLING_RATE, output_path=f"{OUTPUT_DIR}/temporal_filters.png")
    
    logging.info(f"✓ Visualization complete. Figures saved in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()
