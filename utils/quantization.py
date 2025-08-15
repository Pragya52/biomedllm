"""Quantization utilities for federated learning."""

import torch
import numpy as np
from typing import Tuple, Dict, Any

class Quantizer:
    """Utilities for tensor quantization and dequantization."""
    
    @staticmethod
    def quantize_tensor(
        tensor: torch.Tensor, 
        num_bits: int = 8, 
        signed: bool = True
    ) -> Dict[str, Any]:
        """Quantize a tensor to specified bit width."""
        
        if signed:
            qmin = -(2 ** (num_bits - 1))
            qmax = 2 ** (num_bits - 1) - 1
        else:
            qmin = 0
            qmax = 2 ** num_bits - 1
        
        # Calculate scale and zero point
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        if tensor_max == tensor_min:
            scale = torch.tensor(1.0, dtype=tensor.dtype, device=tensor.device)
            zero_point = torch.tensor(qmin, dtype=torch.int8, device=tensor.device)
        else:
            scale = (tensor_max - tensor_min) / (qmax - qmin)
            zero_point = qmin - torch.round(tensor_min / scale)
            zero_point = torch.clamp(zero_point, qmin, qmax)
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)
        
        if signed:
            quantized = quantized.to(torch.int8)
        else:
            quantized = quantized.to(torch.uint8)
        
        return {
            'quantized_tensor': quantized,
            'scale': scale,
            'zero_point': zero_point,
            'original_shape': tensor.shape,
            'original_dtype': tensor.dtype
        }
    
    @staticmethod
    def dequantize_tensor(quantization_data: Dict[str, Any]) -> torch.Tensor:
        """Dequantize a tensor from quantization data."""
        quantized = quantization_data['quantized_tensor']
        scale = quantization_data['scale']
        zero_point = quantization_data['zero_point']
        original_dtype = quantization_data['original_dtype']
        
        # Dequantize
        dequantized = scale * (quantized.float() - zero_point.float())
        return dequantized.to(original_dtype)
