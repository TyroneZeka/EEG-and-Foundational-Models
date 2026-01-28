"""
Smart model loader - infers architecture from checkpoint
"""

import torch
import logging

logger = logging.getLogger(__name__)

def infer_architecture_from_checkpoint(checkpoint_path):
    """Infer EEGNet architecture parameters from saved checkpoint."""
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract architecture info from layer shapes
    info = {}
    
    # From depthwise layer: (F1*D, 1, n_channels, 1)
    if 'depthwise.weight' in checkpoint:
        depthwise_shape = checkpoint['depthwise.weight'].shape
        info['F1_D'] = depthwise_shape[0]  # F1 * D
        info['n_channels'] = depthwise_shape[2]
    
    # From final linear layer: (n_classes, flattened_features)
    if 'fc.weight' in checkpoint:
        fc_shape = checkpoint['fc.weight'].shape
        info['n_classes'] = fc_shape[0]
        info['fc_input_dim'] = fc_shape[1]
    elif 'classifier.weight' in checkpoint:
        fc_shape = checkpoint['classifier.weight'].shape
        info['n_classes'] = fc_shape[0]
        info['fc_input_dim'] = fc_shape[1]
    
    # From batchnorm2 (after depthwise): should match F1*D channels
    if 'batchnorm2.weight' in checkpoint:
        info['D_multiplied'] = checkpoint['batchnorm2.weight'].shape[0]
    
    return info, checkpoint

def get_eegnet_params_from_checkpoint(checkpoint_path):
    """Return the correct EEGNet parameters for loading a model from checkpoint."""
    
    info, checkpoint = infer_architecture_from_checkpoint(checkpoint_path)
    
    # F1*D is known from depthwise layer
    F1_D = info.get('F1_D', 64)
    
    # Infer F1 and D from common patterns
    if F1_D == 16:
        F1, D = 8, 2
    elif F1_D == 32:
        F1, D = 16, 2
    elif F1_D == 64:
        F1, D = 16, 4
    else:
        # Fallback: assume F1=16
        F1 = 16
        D = F1_D // 16
    
    # Get actual n_channels from depthwise kernel (it's the kernel height)
    n_channels_actual = info.get('n_channels', 22)
    
    # Try to infer n_samples from the flattened FC layer size
    # This requires computing: how many features result from pooling?
    # Formula: features = F2 * (n_samples / (4 * 8)) after the pooling operations
    # So: n_samples ≈ fc_input_dim / F2 * 32
    fc_input_dim = info.get('fc_input_dim', 1120)
    F2 = 32
    
    params = {
        'n_channels': n_channels_actual,
        'n_classes': info.get('n_classes', 4),
        'n_samples': 1000,  # Default; will be adjusted below if we can infer it
        'F1': F1,
        'F2': F2,
        'D': D,
        'fc_input_dim': fc_input_dim,  # For verification
    }
    
    logger.info(f"Inferred architecture from checkpoint:")
    logger.info(f"  n_channels: {params['n_channels']}")
    logger.info(f"  n_classes: {params['n_classes']}")
    logger.info(f"  F1: {params['F1']}, F2: {params['F2']}, D: {params['D']}")
    logger.info(f"  FC input features: {fc_input_dim}")
    
    return params
