"""
NeRF model architecture implementation.

This module contains the neural network architectures for the NeRF model,
including position encoding and the MLP-based NeRF network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict, Union


class PositionalEncoding(nn.Module):
    """
    Positional encoding for NeRF input coordinates.
    Applies sin/cos encoding at different frequencies.
    """
    
    def __init__(self, num_freqs: int, include_input: bool = True):
        """
        Initialize positional encoding.
        
        Args:
            num_freqs: Number of frequency bands to use
            include_input: Whether to include the original input
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.funcs = [torch.sin, torch.cos]
        
        # Frequency bands: 2^0, 2^1, 2^2, ..., 2^(num_freqs-1)
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs-1, num_freqs)
        
        # Output dimensions for input of dimension D
        self.out_dim = 0
        if include_input:
            self.out_dim += 1  # original values
        self.out_dim += 2 * num_freqs  # sin and cos encoding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to input.
        
        Args:
            x: Input tensor of shape (..., D)
            
        Returns:
            Encoded tensor of shape (..., D*self.out_dim)
        """
        # Original dimensions
        orig_shape = list(x.shape)
        # Prepare output
        out_shape = orig_shape[:-1] + [orig_shape[-1] * self.out_dim]
        encoded = torch.zeros(out_shape, device=x.device)
        
        # Flatten to simplify encoding
        x_flat = x.reshape(-1, orig_shape[-1])
        encoded_flat = encoded.reshape(-1, out_shape[-1])
        
        # Add original values if specified
        if self.include_input:
            encoded_flat[..., :orig_shape[-1]] = x_flat
            cur_dim = orig_shape[-1]
        else:
            cur_dim = 0
        
        # Apply sin/cos at each frequency
        for freq in self.freq_bands:
            for func in self.funcs:
                encoded_flat[..., cur_dim:cur_dim+orig_shape[-1]] = func(x_flat * freq)
                cur_dim += orig_shape[-1]
        
        # Reshape back to original dimensions
        encoded = encoded_flat.reshape(out_shape)
        return encoded


class NeRF(nn.Module):
    """Neural Radiance Fields (NeRF) model."""
    
    def __init__(self,
                 pos_encoding_freqs: int = 10,
                 dir_encoding_freqs: int = 4,
                 hidden_dim: int = 256,
                 num_layers: int = 8,
                 skip_connections: List[int] = [4]):
        """
        Initialize NeRF model.
        
        Args:
            pos_encoding_freqs: Number of frequency bands for position encoding
            dir_encoding_freqs: Number of frequency bands for direction encoding
            hidden_dim: Number of neurons in hidden layers
            num_layers: Number of hidden layers
            skip_connections: Layers with skip connections
        """
        super().__init__()
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(num_freqs=pos_encoding_freqs)
        self.dir_encoding = PositionalEncoding(num_freqs=dir_encoding_freqs)
        
        # Input dimensions after encoding
        pos_encoded_dim = 3 * self.pos_encoding.out_dim  # x, y, z coordinates
        dir_encoded_dim = 3 * self.dir_encoding.out_dim  # viewing direction
        
        # MLP layers for density prediction
        self.skip_connections = skip_connections
        self.layers_xyz = nn.ModuleList()
        
        # First layer takes encoded position as input
        self.layers_xyz.append(nn.Linear(pos_encoded_dim, hidden_dim))
        
        # Hidden layers with skip connections
        for i in range(num_layers - 1):
            if i in skip_connections:
                # Skip connection: concat input to current activations
                self.layers_xyz.append(nn.Linear(hidden_dim + pos_encoded_dim, hidden_dim))
            else:
                self.layers_xyz.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Density prediction (sigma)
        self.density_layer = nn.Linear(hidden_dim, 1)
        
        # Feature vector for color prediction
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
        
        # Color prediction layers
        self.color_layer = nn.Sequential(
            nn.Linear(hidden_dim + dir_encoded_dim, hidden_dim // 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()  # Colors are in [0, 1]
        )

    def forward(self, 
               x: torch.Tensor, 
               d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the NeRF model.
        
        Args:
            x: 3D positions, shape (..., 3)
            d: Viewing directions, shape (..., 3)
            
        Returns:
            Tuple of (rgb, sigma):
                - rgb: RGB colors, shape (..., 3)
                - sigma: Volume density, shape (..., 1)
        """
        # Encode inputs
        x_encoded = self.pos_encoding(x)
        d_encoded = self.dir_encoding(d)
        
        # Process position through MLP with skip connections
        h = x_encoded
        for i, layer in enumerate(self.layers_xyz):
            if i in self.skip_connections:
                h = torch.cat([h, x_encoded], dim=-1)
            h = F.relu(layer(h))
        
        # Density prediction
        sigma = F.relu(self.density_layer(h))
        
        # Feature vector for color
        feature = self.feature_layer(h)
        
        # Predict color based on feature and viewing direction
        h_rgb = torch.cat([feature, d_encoded], dim=-1)
        rgb = self.color_layer(h_rgb)
        
        return rgb, sigma


class NeRFModel(nn.Module):
    """
    Complete NeRF model with coarse and fine networks.
    This implementats the hierarchical sampling approach from the NeRF paper.
    """
    
    def __init__(self,
                pos_encoding_freqs: int = 10,
                dir_encoding_freqs: int = 4,
                hidden_dim: int = 256,
                num_layers: int = 8):
        """
        Initialize the complete NeRF model.
        
        Args:
            pos_encoding_freqs: Number of frequency bands for position encoding
            dir_encoding_freqs: Number of frequency bands for direction encoding
            hidden_dim: Number of neurons in hidden layers
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        # Create coarse and fine networks
        self.nerf_coarse = NeRF(
            pos_encoding_freqs=pos_encoding_freqs,
            dir_encoding_freqs=dir_encoding_freqs,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        self.nerf_fine = NeRF(
            pos_encoding_freqs=pos_encoding_freqs,
            dir_encoding_freqs=dir_encoding_freqs,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
    
    def forward(self, 
               x_coarse: torch.Tensor, 
               d_coarse: torch.Tensor,
               x_fine: Optional[torch.Tensor] = None,
               d_fine: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both coarse and fine networks.
        
        Args:
            x_coarse: Coarse sample positions, shape (..., 3)
            d_coarse: Coarse viewing directions, shape (..., 3)
            x_fine: Fine sample positions (optional), shape (..., 3)
            d_fine: Fine viewing directions (optional), shape (..., 3)
            
        Returns:
            Dictionary with:
                - 'rgb_coarse': Coarse RGB values
                - 'sigma_coarse': Coarse density values
                - 'rgb_fine': Fine RGB values (if fine inputs provided)
                - 'sigma_fine': Fine density values (if fine inputs provided)
        """
        # Process coarse samples
        rgb_coarse, sigma_coarse = self.nerf_coarse(x_coarse, d_coarse)
        
        outputs = {
            'rgb_coarse': rgb_coarse,      # (..., 3)
            'sigma_coarse': sigma_coarse,  # (..., 1)
        }
        
        # Process fine samples if provided
        if x_fine is not None and d_fine is not None:
            rgb_fine, sigma_fine = self.nerf_fine(x_fine, d_fine)
            outputs['rgb_fine'] = rgb_fine
            outputs['sigma_fine'] = sigma_fine
        
        return outputs
