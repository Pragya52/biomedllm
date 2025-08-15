import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class ClientConfig:
    """Configuration for federated learning clients using real medical QA data."""
    
    # Model configuration - Full BioMedLM
    model_name: str = "stanford-crfm/BioMedLM"
    model_revision: str = "main"
    split_layer: int = 16  # Split at middle of 32 layers for better balance
    embedding_dim: int = 4096
    local_layers: int = 4  # Deeper local processing
    
    # Privacy configuration
    gaussian_noise_std: float = 0.01  # Lower noise for QA accuracy
    quantization_bits: int = 8
    dp_epsilon: float = 1.0  # Differential privacy budget
    
    # Training configuration
    learning_rate: float = 5e-5  # Lower LR for large model
    batch_size: int = 4  # Smaller batches for memory efficiency
    local_epochs: int = 2
    max_sequence_length: int = 1024  # Longer for QA contexts
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Medical QA specific
    qa_format: str = "extractive"  # extractive, generative, or multiple_choice
    max_answer_length: int = 128
    context_window: int = 512
    
    # Communication configuration
    server_url: str = "http://localhost:5000"
    client_id: str = "client_1"
    timeout: int = 300  # Longer timeout for large models
    
    # Dataset configuration
    dataset_name: str = "medqa"  # medqa, pubmedqa, bioasq
    data_split: str = "train"
    num_samples: Optional[int] = None  # Use all available data
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  # Enable for memory efficiency
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
