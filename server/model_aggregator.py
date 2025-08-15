"""Model aggregation for federated learning."""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from threading import Lock
import logging

logger = logging.getLogger(__name__)

class FedAvgAggregator:
    """Federated Averaging aggregator for client weights."""
    
    def __init__(self, num_clients: int, min_clients: int = 2):
        self.num_clients = num_clients
        self.min_clients = min_clients
        self.client_weights = {}
        self.lock = Lock()
        self.round_count = 0
        
    def add_client_weights(self, client_id: str, weights: Dict[str, List]) -> Optional[Dict[str, List]]:
        """Add client weights and return aggregated weights if ready."""
        
        with self.lock:
            # Convert lists back to tensors for aggregation
            tensor_weights = {
                name: torch.tensor(weight_list) 
                for name, weight_list in weights.items()
            }
            
            self.client_weights[client_id] = tensor_weights
            
            logger.info(f"Received weights from {client_id}. "
                       f"Total clients: {len(self.client_weights)}/{self.num_clients}")
            
            # Check if we have enough clients to aggregate
            if len(self.client_weights) >= self.min_clients:
                aggregated = self._aggregate_weights()
                self.client_weights.clear()  # Reset for next round
                self.round_count += 1
                return aggregated
            
            return None
    
    def _aggregate_weights(self) -> Dict[str, List]:
        """Perform FedAvg aggregation."""
        
        logger.info(f"Aggregating weights from {len(self.client_weights)} clients")
        
        # Get all weight names from first client
        client_ids = list(self.client_weights.keys())
        weight_names = list(self.client_weights[client_ids[0]].keys())
        
        aggregated_weights = {}
        
        for weight_name in weight_names:
            # Collect weights from all clients
            client_weight_list = [
                self.client_weights[client_id][weight_name] 
                for client_id in client_ids
            ]
            
            # Stack and average
            stacked_weights = torch.stack(client_weight_list)
            avg_weight = torch.mean(stacked_weights, dim=0)
            
            # Convert back to list for JSON serialization
            aggregated_weights[weight_name] = avg_weight.tolist()
        
        logger.info(f"Weights aggregated successfully for round {self.round_count}")
        return aggregated_weights
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get statistics about the aggregation process."""
        
        return {
            'total_rounds': self.round_count,
            'clients_participating': len(self.client_weights),
            'min_clients_required': self.min_clients,
            'total_clients_expected': self.num_clients
        }
