"""
EEGNet implementation - Canonical architecture for EEG classification.
Based on: Lawhern et al. (2018) "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class EEGNet(nn.Module):
    """
    EEGNet: Compact CNN for EEG Classification.
    
    Architecture:
    1. Initial temporal convolution (1, C) -> (F1, C)
    2. Spatial convolution (depthwise) (F1, 1) -> (F1, 1)
    3. Depthwise separable convolution (1, 1) and (C, 1) -> (F2, C)
    4. Classification head
    """
    
    def __init__(self,
                 n_channels,
                 n_classes,
                 n_samples,
                 F1=8,
                 F2=16,
                 D=2,
                 drop_rate=0.5,
                 kernel_length=64):
        """
        Args:
            n_channels: Number of EEG channels
            n_classes: Number of classification classes
            n_samples: Number of time samples per trial
            F1: Number of temporal filters
            F2: Number of pointwise filters
            D: Depth multiplier for depthwise convolution
            drop_rate: Dropout rate
            kernel_length: Length of temporal convolution kernel
        """
        super(EEGNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_samples = n_samples
        
        # Block 1: Temporal Convolution
        self.conv1 = nn.Conv2d(
            1, F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        # Block 2: Spatial Convolution (Depthwise)
        self.depthwise = nn.Conv2d(
            F1, F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
            padding=0
        )
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation = nn.ELU(alpha=1.0)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(drop_rate)
        
        # Block 3: Depthwise Separable Convolution
        self.separable_depth = nn.Conv2d(
            F1 * D, F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=False
        )
        self.separable_point = nn.Conv2d(
            F1 * D, F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(drop_rate)
        
        # Calculate flattened size
        self.flatten_size = self._calculate_flatten_size()
        
        # Classification head
        self.fc = nn.Linear(self.flatten_size, n_classes)
        
        logger.info(f"EEGNet initialized: {n_channels} channels, {n_classes} classes")
        logger.info(f"Flatten size: {self.flatten_size}")
    
    def _calculate_flatten_size(self):
        """Calculate the flattened feature size after convolutions."""
        x = torch.zeros(1, 1, self.n_channels, self.n_samples)
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        return int(np.prod(x.shape[1:]))
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch_size, n_channels, n_samples)
        
        Returns:
            logits: (batch_size, n_classes)
        """
        # Add channel dimension: (batch, 1, channels, samples)
        x = x.unsqueeze(1)
        
        # Block 1: Temporal
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        # Block 2: Spatial (Depthwise)
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 3: Depthwise Separable
        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def get_intermediate_features(self, x, layer='after_block2'):
        """
        Extract intermediate features for visualization (t-SNE, gradients).
        
        Args:
            x: input tensor
            layer: 'block1_out', 'block2_out', or 'output'
        
        Returns:
            features at the specified layer
        """
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        if layer == 'block1_out':
            return x.view(x.size(0), -1)
        
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        if layer == 'block2_out':
            return x.view(x.size(0), -1)
        
        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)
        return x


def create_eegnet(n_channels, n_classes, n_samples, **kwargs):
    """Factory function to create EEGNet model."""
    return EEGNet(n_channels, n_classes, n_samples, **kwargs)


# For shape calculation
import numpy as np
