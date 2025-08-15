import torch
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ServerConfig:
    """Configuration for federated learning server."""
    
    # Model configuration - Full BioMedLM
    model_name: str = "stanford-crfm/BioMedLM"
    model_revision: str = "main"
    total_layers: int = 32
    split_layer: int = 16  # Server handles layers 16-32
    embedding_dim: int = 4096
    
    # Federated learning configuration
    num_clients: int = 3
    fedavg_frequency: int = 3  # More frequent aggregation for QA
    min_clients_for_aggregation: int = 2
    
    # Training configuration
    global_rounds: int = 50
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    
    # QA Task configuration
    qa_format: str = "extractive"
    num_choices: int = 4  # For multiple choice QA
    max_answer_length: int = 128
    
    # Privacy configuration
    quantization_bits: int = 8
    secure_aggregation: bool = True
    
    # Communication configuration
    port: int = 5000
    host: str = "0.0.0.0"
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    
    # Logging configuration
    log_frequency: int = 5
    save_model_frequency: int = 10
    model_save_path: str = "./saved_models"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
