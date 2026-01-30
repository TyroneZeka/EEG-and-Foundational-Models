import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# --- Setup ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.biot import BIOTClassifier
from data.load_data import DataLoader as EEGDataLoader
from data.preprocessing import EEGPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global list to store captured attention maps
attention_maps_by_layer = {}

def visualize_attention(model, data_loader, device, output_dir):
    logging.info("Visualizing self-attention maps...")
    
    # --- THIS IS THE FINAL, CORRECT METHOD ---
    
    # 1. Find all the 'Attention' modules we want to analyze
    attention_layers = []
    for name, module in model.named_modules():
        if 'Attention' in module.__class__.__name__ and 'transformer' in name:
            attention_layers.append((name, module))
            
    if not attention_layers:
        logging.error("Could not find any 'Attention' modules. Cannot visualize.")
        return

    # 2. Monkey-patch their forward methods to force return_attn=True
    original_forwards = {}
    for name, module in attention_layers:
        # Save the original forward method
        original_forwards[name] = module.forward
        
        # Create and set the new forward method
        def new_forward(self, x, **kwargs):
            kwargs['return_attn'] = True
            return original_forwards[name](x, **kwargs)
            
        module.forward = new_forward.__get__(module, module.__class__)
        logging.info(f"Patched forward method for layer: {name}")

    # 3. Use hooks to capture the output of the now-modified methods
    handles = []
    for name, module in attention_layers:
        # The hook will now receive the (features, attention) tuple
        def create_hook(layer_name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    attention_maps_by_layer[layer_name] = output[1].detach().cpu().numpy()
            return hook
        handle = module.register_forward_hook(create_hook(name))
        handles.append(handle)

    # 4. Run one batch to trigger everything
    model.eval()
    with torch.no_grad():
        X_batch, _ = next(iter(data_loader))
        _ = model(X_batch.to(device))

    # 5. Clean up: Restore original forward methods and remove hooks
    for name, module in attention_layers:
        module.forward = original_forwards[name]
    for handle in handles:
        handle.remove()
        
    # --- END OF THE FIX ---

    if not attention_maps_by_layer:
        logging.error("Monkey-patching and hooks were applied, but no attention maps were captured.")
        return

    # Plotting
    layers_to_plot_indices = [0, len(attention_layers) // 2, len(attention_layers) - 1]
    layers_to_plot_names = [attention_layers[i][0] for i in layers_to_plot_indices]
    
    fig, axes = plt.subplots(1, len(layers_to_plot_indices), figsize=(8 * len(layers_to_plot_indices), 6))
    if len(layers_to_plot_indices) == 1: axes = [axes]
    
    for ax, name in zip(axes, layers_to_plot_names):
        attn_map = attention_maps_by_layer.get(name)
        if attn_map is not None:
            avg_map = np.mean(attn_map, axis=(0, 1))
            im = ax.imshow(avg_map, cmap='viridis')
            ax.set_title(f"Layer: ...{name.split('.')[-3]}", fontsize=12)
            fig.colorbar(im, ax=ax, shrink=0.8)
    
    plt.suptitle('Self-Attention Maps from Different Layers', fontsize=16)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'attention_maps.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"✓ Saved attention map visualization to {output_path}")

def main():
    # --- USER CONFIGURATION ---
    DATASET_NAME = 'BCI_IV_2a'
    MODEL_PATH = f'models/biot/{DATASET_NAME}_fine_tune.pth'
    OUTPUT_DIR = f'analysis/biot_attention/{DATASET_NAME}'
    # --- END OF CONFIGURATION ---

    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found: {MODEL_PATH}")
        return

    data_loader_manager = EEGDataLoader()
    preprocessor = EEGPreprocessor(sampling_rate=250)
    data = getattr(data_loader_manager, 'load_bci_2a')()
    X, y = preprocessor.preprocess(data['X'], data['y'])
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    model_config = {'n_channels': X.shape[1], 'n_classes': len(np.unique(y)), 'emb_size': 256, 'heads': 8, 'depth': 4, 'n_fft': 200, 'hop_length': 20}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BIOTClassifier(**model_config).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    visualize_attention(model, data_loader, device, OUTPUT_DIR)

if __name__ == "__main__":
    main()