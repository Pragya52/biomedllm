"""Communication utilities for federated learning."""

import requests
import json
import logging
from typing import Dict, Any, Optional, List
import numpy as np
import torch

logger = logging.getLogger(__name__)

class ClientCommunicator:
    """Handles communication between client and server."""
    
    def __init__(self, server_url: str, client_id: str, timeout: int = 300):
        self.server_url = server_url
        self.client_id = client_id
        self.timeout = timeout
        
    def send_smashed_data(
        self, 
        privacy_data: Dict[str, Any], 
        batch_data: Dict[str, Any],
        qa_format: str
    ) -> Optional[Dict[str, Any]]:
        """Send smashed data to server and get predictions."""
        
        try:
            # Convert tensors to serializable format
            serialized_privacy_data = self._serialize_privacy_data(privacy_data)
            serialized_batch_data = self._serialize_batch_data(batch_data)
            
            # Prepare payload
            payload = {
                'client_id': self.client_id,
                'smashed_data': serialized_privacy_data,
                'batch_data': serialized_batch_data,
                'qa_format': qa_format
            }
            
            # Send request
            response = requests.post(
                f"{self.server_url}/smashed_data",
                json=payload,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Server returned status code: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Communication error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in communication: {e}")
            return None
    
    def send_weights_for_fedavg(self, weights: Dict[str, List]) -> Optional[Dict[str, Any]]:
        """Send weights to server for federated averaging."""
        
        try:
            payload = {
                'client_id': self.client_id,
                'weights': weights
            }
            
            response = requests.post(
                f"{self.server_url}/fedavg",
                json=payload,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"FedAvg request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error in FedAvg communication: {e}")
            return None
    
    def check_server_status(self) -> bool:
        """Check if server is available."""
        try:
            response = requests.get(
                f"{self.server_url}/status",
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    def _serialize_privacy_data(self, privacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert privacy data tensors to serializable format."""
        
        serialized = {}
        for key, value in privacy_data.items():
            if isinstance(value, torch.Tensor):
                serialized[key] = value.cpu().numpy().tolist()
            elif isinstance(value, tuple):
                serialized[key] = list(value)
            else:
                serialized[key] = value
        
        return serialized
    
    def _serialize_batch_data(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert batch data tensors to serializable format."""
        
        serialized = {}
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                serialized[key] = value.cpu().numpy().tolist()
            elif isinstance(value, (list, tuple)):
                serialized[key] = list(value)
            else:
                serialized[key] = value
        
        return serialized
