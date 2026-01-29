import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging

# --- Setup Paths and Logging ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.eegnet import EEGNet
from data.load_data import DataLoader as EEGDataLoader
from data.preprocessing import EEGPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Plotting Function ---
def plot_tsne(features, labels, title, output_path, class_names):
    """Computes and plots the t-SNE visualization."""
    logging.info(f"Running t-SNE for '{title}'...")
    if len(features) < 5:
        logging.warning("Not enough samples for t-SNE. Skipping.")
        return
    
    features_scaled = StandardScaler().fit_transform(features)
    # Adjust perplexity if there are fewer samples than the default, to prevent errors
    perplexity = min(30, len(features_scaled) - 1)
    
    tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, max_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(features_scaled)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.get_cmap('jet', len(class_names))
    for i, name in enumerate(class_names):
        idxs = np.where(labels == i)[0]
        if len(idxs) > 0:
            ax.scatter(tsne_results[idxs, 0], tsne_results[idxs, 1], color=colors(i), label=name, alpha=0.7, s=50)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"✓ Saved t-SNE plot to {output_path}")

# --- Main Execution Logic ---
def main():
    """
    Main function to loop through all datasets and folds, generating all multi-stage t-SNE visualizations.
    """
    # --- MASTER CONFIGURATION ---
    BASE_MODEL_PATH = 'experiments/task1_eegnet'
    FINAL_FIGURES_DIR = 'analysis/multistage_tsne'
    
    # Model hyperparameters (MUST match all trained models)
    MODEL_HYPERPARAMS = {'F1': 16, 'D': 4, 'F2': 32}

    # Dataset-specific configurations
    DATASETS_CONFIG = {
        'BCI_IV_2a': {
            'loader_func': 'load_bci_2a',
            'n_classes': 4,
            'class_names': ['L Hand', 'R Hand', 'Feet', 'Tongue']
        },
        'BCI_IV_2b': {
            'loader_func': 'load_bci_2b',
            'n_classes': 2,
            'class_names': ['L Hand', 'R Hand']
        },
        'PhysioNet_MI': {
            'loader_func': 'load_physionet_mi',
            'n_classes': 3,
            'class_names': ['T0', 'T1', 'T2'] # Using actual event names from logs
        }
    }
    
    N_SAMPLES_FOR_TSNE = 1000 # Keep this reasonably low to run quickly for all subjects
    # --- END OF CONFIGURATION ---

    data_loader = EEGDataLoader()
    preprocessor = EEGPreprocessor(sampling_rate=250)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for name, config in DATASETS_CONFIG.items():
        logging.info(f"\n{'='*80}\nProcessing Dataset: {name}\n{'='*80}")
        
        # Load the full dataset once
        try:
            loader = getattr(data_loader, config['loader_func'])
            data = loader()
            X, y = preprocessor.preprocess(data['X'], data['y'])
            subjects = np.unique(data['metadata']['subject'])
        except Exception as e:
            logging.error(f"Failed to load or preprocess {name}. Skipping. Error: {e}")
            continue

        for subject in subjects:
            fold_num = int(subject)
            logging.info(f"\n--- Analyzing {name}, Fold/Subject: {fold_num} ---")
            
            # Define paths for this specific fold
            model_path = Path(BASE_MODEL_PATH) / name / f"fold_{fold_num}_best_model.pth"
            output_dir = Path(FINAL_FIGURES_DIR) / name / f"fold_{fold_num}"
            os.makedirs(output_dir, exist_ok=True)

            if not model_path.exists():
                logging.warning(f"Model not found for {name}, fold {fold_num} at {model_path}. Skipping fold.")
                continue
            
            # Select data for the current subject
            subject_indices = np.where(data['metadata']['subject'] == subject)[0]
            X_subj, y_subj = X[subject_indices], y[subject_indices]
            
            if len(X_subj) == 0:
                logging.warning(f"No data found for subject {fold_num}. Skipping.")
                continue

            # Take a random subset for efficiency
            subset_indices = np.random.choice(len(X_subj), min(len(X_subj), N_SAMPLES_FOR_TSNE), replace=False)
            X_subset, y_subset = X_subj[subset_indices], y_subj[subset_indices]
            
            n_channels = X_subset.shape[1]
            n_samples = X_subset.shape[2]

            # --- STAGE 1: Raw Input ---
            plot_tsne(X_subset.reshape(X_subset.shape[0], -1), y_subset,
                      title=f'Stage 1: Raw Input Data ({name} - Subject {fold_num})',
                      output_path=output_dir / "stage1_raw_input.png",
                      class_names=config['class_names'])
            
            # --- Load Model and run Stages 2 & 3 ---
            try:
                model = EEGNet(
                    n_channels=n_channels, n_classes=config['n_classes'], n_samples=n_samples,
                    **MODEL_HYPERPARAMS
                ).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()

                with torch.no_grad():
                    X_tensor = torch.from_numpy(X_subset).float().to(device)
                    hidden_features = model.get_intermediate_features(X_tensor, layer='output').cpu().numpy()
                    output_logits = model(X_tensor).cpu().numpy()

                # Plot Stage 2
                plot_tsne(hidden_features, y_subset,
                          title=f'Stage 2: Hidden Features ({name} - Subject {fold_num})',
                          output_path=output_dir / "stage2_hidden_features.png",
                          class_names=config['class_names'])

                # Plot Stage 3
                plot_tsne(output_logits, y_subset,
                          title=f'Stage 3: Output Logits ({name} - Subject {fold_num})',
                          output_path=output_dir / "stage3_output_logits.png",
                          class_names=config['class_names'])
            except Exception as e:
                logging.error(f"Failed to generate plots for {name}, fold {fold_num}. Error: {e}")
                continue

if __name__ == "__main__":
    main()
