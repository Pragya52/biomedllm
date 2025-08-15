"""Federated learning server for medical QA with BioMedLM."""

import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, jsonify
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from threading import Lock
import json

from ..config.server_config import ServerConfig
from ..models.biomedlm_wrapper import FullBioMedLMWrapper
from ..models.split_model import ServerQAModel
from ..models.privacy_layers import PrivacyManager
from .model_aggregator import FedAvgAggregator
from ..utils.metrics import QAMetricsTracker

logger = logging.getLogger(__name__)

class FederatedQAServer:
    """Federated learning server for medical QA."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup logging
        logger.setLevel(logging.INFO)
        
        # Initialize BioMedLM wrapper
        self.biomedlm_wrapper = FullBioMedLMWrapper(
            model_name=config.model_name,
            model_revision=config.model_revision,
            split_layer=config.split_layer,
            qa_format=config.qa_format
        )
        
        # Initialize server model
        self.server_model = ServerQAModel(self.biomedlm_wrapper).to(self.device)
        
        # Privacy manager for dequantization
        self.privacy_manager = PrivacyManager(
            noise_std=0.0,  # Server doesn't add noise
            num_bits=config.quantization_bits
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.server_model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # FedAvg aggregator
        self.aggregator = FedAvgAggregator(
            num_clients=config.num_clients,
            min_clients=config.min_clients_for_aggregation
        )
        
        # Metrics tracking
        self.metrics = QAMetricsTracker(config.qa_format)
        
        # Thread safety
        self.lock = Lock()
        
        # Client management
        self.connected_clients = set()
        self.round_count = 0
        
        # Flask app for communication
        self.app = Flask(__name__)
        self.setup_routes()
        
        logger.info("Federated QA server initialized")
    
    def setup_routes(self):
        """Setup Flask routes for client communication."""
        
        @self.app.route('/smashed_data', methods=['POST'])
        def receive_smashed_data():
            """Receive smashed data from clients."""
            try:
                data = request.get_json()
                client_id = data['client_id']
                smashed_data = data['smashed_data']
                batch_data = data['batch_data']
                qa_format = data['qa_format']
                
                # Process smashed data
                response = self.process_smashed_data(client_id, smashed_data, batch_data, qa_format)
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Error processing smashed data: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/fedavg', methods=['POST'])
        def receive_weights_for_fedavg():
            """Receive weights from clients for federated averaging."""
            try:
                data = request.get_json()
                client_id = data['client_id']
                weights = data['weights']
                
                # Add to aggregator
                aggregated_weights = self.aggregator.add_client_weights(client_id, weights)
                
                response = {'status': 'received'}
                if aggregated_weights is not None:
                    response['aggregated_weights'] = aggregated_weights
                    self.round_count += 1
                    logger.info(f"FedAvg completed for round {self.round_count}")
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Error in FedAvg: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/status', methods=['GET'])
        def get_status():
            """Get server status."""
            return jsonify({
                'connected_clients': len(self.connected_clients),
                'round_count': self.round_count,
                'server_config': self.config.to_dict()
            })
    
    def process_smashed_data(
        self, 
        client_id: str, 
        smashed_data: Dict[str, Any], 
        batch_data: Dict[str, Any],
        qa_format: str
    ) -> Dict[str, Any]:
        """Process smashed data from client and return predictions."""
        
        try:
            with self.lock:
                # Reconstruct tensor from privacy data
                privacy_data = {
                    'data': torch.tensor(smashed_data['data'], dtype=torch.int8).to(self.device),
                    'scale': torch.tensor(smashed_data['scale']).to(self.device),
                    'zero_point': torch.tensor(smashed_data['zero_point'], dtype=torch.int8).to(self.device),
                    'shape': tuple(smashed_data['shape']),
                    'dtype': getattr(torch, smashed_data['dtype'].split('.')[-1])
                }
                
                # Dequantize
                hidden_states = self.privacy_manager.remove_privacy(privacy_data)
                hidden_states.requires_grad_(True)
                
                # Prepare batch data for server
                server_batch = self._prepare_server_batch(batch_data, qa_format)
                
                # Server forward pass
                self.server_model.train()
                outputs = self.server_model(hidden_states, **server_batch)
                
                loss = outputs.get('loss', torch.tensor(0.0))
                
                # Backward pass
                if loss.requires_grad:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.server_model.parameters(), 1.0)
                    self.optimizer.step()
                
                # Prepare response based on QA format
                response = self._prepare_response(outputs, qa_format, client_id)
                
                # Add to connected clients
                self.connected_clients.add(client_id)
                
                return response
                
        except Exception as e:
            logger.error(f"Error processing smashed data from {client_id}: {e}")
            raise
    
    def _prepare_server_batch(self, batch_data: Dict[str, Any], qa_format: str) -> Dict[str, Any]:
        """Prepare batch data for server processing."""
        
        server_batch = {}
        
        if qa_format == "multiple_choice":
            if 'labels' in batch_data:
                server_batch['labels'] = torch.tensor(batch_data['labels']).to(self.device)
            if 'choice_input_ids' in batch_data:
                server_batch['choice_input_ids'] = torch.tensor(batch_data['choice_input_ids']).to(self.device)
                
        elif qa_format == "extractive":
            if 'start_positions' in batch_data:
                server_batch['start_positions'] = torch.tensor(batch_data['start_positions']).to(self.device)
            if 'end_positions' in batch_data:
                server_batch['end_positions'] = torch.tensor(batch_data['end_positions']).to(self.device)
                
        elif qa_format == "generative":
            if 'labels' in batch_data:
                server_batch['labels'] = torch.tensor(batch_data['labels']).to(self.device)
        
        return server_batch
    
    def _prepare_response(self, outputs: Dict[str, torch.Tensor], qa_format: str, client_id: str) -> Dict[str, Any]:
        """Prepare response based on QA format."""
        
        response = {'client_id': client_id, 'qa_format': qa_format}
        
        if 'loss' in outputs:
            response['loss'] = outputs['loss'].item()
        
        if qa_format == "multiple_choice":
            if 'logits' in outputs:
                response['logits'] = outputs['logits'].detach().cpu().numpy().tolist()
                
        elif qa_format == "extractive":
            if 'start_logits' in outputs:
                response['start_logits'] = outputs['start_logits'].detach().cpu().numpy().tolist()
            if 'end_logits' in outputs:
                response['end_logits'] = outputs['end_logits'].detach().cpu().numpy().tolist()
                
        elif qa_format == "generative":
            if 'logits' in outputs:
                response['logits'] = outputs['logits'].detach().cpu().numpy().tolist()
        
        # Log metrics
        if 'loss' in response:
            self.metrics.log_metrics(
                round_num=self.round_count,
                metrics={f'server_loss_{client_id}': response['loss']}
            )
        
        return response
    
    def start_server(self):
        """Start the federated learning server."""
        logger.info(f"Starting federated QA server on {self.config.host}:{self.config.port}")
        logger.info(f"QA Format: {self.config.qa_format}")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Split Layer: {self.config.split_layer}")
        
        self.app.run(
            host=self.config.host,
            port=self.config.port,
            debug=False,
            threaded=True
        )
