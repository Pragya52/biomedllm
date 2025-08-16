"""Privacy-preserving layers for federated learning."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any

class GaussianNoiseLayer(nn.Module):
    """Adds Gaussian noise for differential privacy."""
    
    def __init__(self, noise_std: float = 0.1):
        super().__init__()
        self.noise_std = noise_std
        self.training = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise during training."""
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x
    
    def set_noise_std(self, noise_std: float):
        """Update noise standard deviation."""
        self.noise_std = noise_std

class QuantizationLayer(nn.Module):
    """Quantizes tensors to int8 for communication efficiency."""
    
    def __init__(self, num_bits: int = 8):
        super().__init__()
        self.num_bits = num_bits
        self.qmin = -(2 ** (num_bits - 1))
        self.qmax = 2 ** (num_bits - 1) - 1
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize tensor to int8."""
        # Calculate scale and zero_point
        x_min, x_max = x.min(), x.max()
        
        # Avoid division by zero
        if x_max == x_min:
            scale = torch.tensor(1.0, dtype=x.dtype, device=x.device)
            zero_point = torch.tensor(0, dtype=torch.int8, device=x.device)
        else:
            scale = (x_max - x_min) / (self.qmax - self.qmin)
            zero_point = self.qmin - torch.round(x_min / scale)
            zero_point = torch.clamp(zero_point, self.qmin, self.qmax).to(torch.int8)
        
        # Quantize
        x_q = torch.round(x / scale + zero_point)
        x_q = torch.clamp(x_q, self.qmin, self.qmax).to(torch.int8)
        
        return x_q, scale, zero_point
    
    def dequantize(
        self, 
        x_q: torch.Tensor, 
        scale: torch.Tensor, 
        zero_point: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize int8 tensor back to float."""
        return scale * (x_q.float() - zero_point.float())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (no quantization during training)."""
        return x

class PrivacyManager(nn.Module):
    """Manages privacy-preserving transformations."""
    
    def __init__(self, noise_std: float = 0.1, num_bits: int = 8):
        super().__init__()
        self.noise_layer = GaussianNoiseLayer(noise_std)
        self.quantization_layer = QuantizationLayer(num_bits)
    
    def apply_privacy(self, x: torch.Tensor) -> Dict[str, Any]:
        """Apply noise and quantization for privacy."""
        # Add Gaussian noise
        x_noisy = self.noise_layer(x)
        
        # Quantize
        x_q, scale, zero_point = self.quantization_layer.quantize(x_noisy)
        
        return {
            "data": x_q,
            "scale": scale,
            "zero_point": zero_point,
            "shape": x.shape,
            "dtype": str(x.dtype)
        }
    
    def remove_privacy(self, privacy_data: Dict[str, Any]) -> torch.Tensor:
        """Remove quantization (noise cannot be removed)."""
        return self.quantization_layer.dequantize(
            privacy_data["data"],
            privacy_data["scale"],
            privacy_data["zero_point"]
        ).reshape(privacy_data["shape"]).to(privacy_data["dtype"])
