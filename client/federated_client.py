import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import logging
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import json

from config.client_config import ClientConfig
from models.biomedlm_wrapper import FullBioMedLMWrapper
from models.split_model import ClientQAModel
from models.privacy_layers import PrivacyManager
from utils.communication import ClientCommunicator
from utils.metrics import QAMetricsTracker
from data.data_loader import FederatedQADataLoader

logger = logging.getLogger(__name__)

class FederatedQAClient:
    """Federated learning client for medical QA with BioMedLM."""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup logging
        logger.setLevel(logging.INFO)
        
        # Initialize BioMedLM wrapper
        self.biomedlm_wrapper = FullBioMedLMWrapper(
            model_name=config.model_name,
            model_revision=config.model_revision,
            split_layer=config.split_layer,
            qa_format=config.qa_format,
            max_answer_length=config.max_answer_length
        )
        
        # Initialize client model
        self.client_model = ClientQAModel(
            biomedlm_wrapper=self.biomedlm_wrapper,
            local_layers=config.local_layers,
            client_id=config.client_id
        ).to(self.device)
        
        # Privacy manager
        self.privacy_manager = PrivacyManager(
            noise_std=config.gaussian_noise_std,
            num_bits=config.quantization_bits
        ).to(self.device)
        
        # Optimizer with gradient accumulation
        self.optimizer = optim.AdamW(
            self.client_model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Data loader
        self.data_loader = FederatedQADataLoader(config.client_id, config)
        self.train_loader, self.val_loader = self.data_loader.load_client_data()
        
        # Communication
        self.communicator = ClientCommunicator(config.server_url, config.client_id)
        
        # Metrics tracking
        self.metrics = QAMetricsTracker(config.qa_format)
        
        # Loss functions
        self.distillation_criterion = nn.KLDivLoss(reduction='batchmean')
        
        logger.info(f"Initialized federated QA client {config.client_id}")
        logger.info(f"QA Format: {config.qa_format}")
        logger.info(f"Model split at layer {config.split_layer}")
    
    def local_training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one local training step with mixed precision."""
        
        self.client_model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Extract common inputs
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')
        
        with autocast(device_type=self.device.type, enabled=self.scaler is not None):
            
            # Local forward pass for distillation
            batch_copy = batch.copy()
            batch_copy.pop('input_ids', None)
            batch_copy.pop('attention_mask', None)

            batch_for_local = {k: v for k, v in batch.items() if k not in ['input_ids', 'attention_mask']}

            local_outputs = self.client_model.forward_local(
             input_ids=input_ids,
               attention_mask=attention_mask,
                  **batch_for_local
                )
            local_loss = local_outputs.get('loss', torch.tensor(0.0))
            
            # Global path forward pass
            global_hidden = self.client_model.forward_global_path(input_ids, attention_mask)
            
            # Apply privacy preserving transformations
            privacy_data = self.privacy_manager.apply_privacy(global_hidden)
            
            # Send to server and get predictions
            server_response = self.communicator.send_smashed_data(
                privacy_data, batch, self.config.qa_format
            )
            
            distillation_loss = torch.tensor(0.0, device=self.device)
            
            if server_response is not None:
                # Server predictions for distillation
                server_outputs = self._process_server_response(server_response, batch)
                
                # Knowledge distillation loss
                distillation_loss = self._compute_distillation_loss(
                    local_outputs, server_outputs
                )
            
            # Combined loss
            alpha = 0.6  # Weight for local loss
            beta = 0.4   # Weight for distillation loss
            total_loss = alpha * local_loss + beta * distillation_loss
        
        # Backward pass with gradient accumulation
        if self.scaler is not None:
            # Mixed precision backward
            scaled_loss = self.scaler.scale(total_loss / self.config.gradient_accumulation_steps)
            scaled_loss.backward()
        else:
            # Regular backward
            (total_loss / self.config.gradient_accumulation_steps).backward()
        
        return {
            'local_loss': local_loss.item(),
            'distillation_loss': distillation_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def _process_server_response(
        self, 
        server_response: Dict[str, Any], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process server response based on QA format."""
        
        if self.config.qa_format == "multiple_choice":
            logits = torch.tensor(server_response['logits'], device=self.device)
            return {"logits": logits}
            
        elif self.config.qa_format == "extractive":
            start_logits = torch.tensor(server_response['start_logits'], device=self.device)
            end_logits = torch.tensor(server_response['end_logits'], device=self.device)
            return {"start_logits": start_logits, "end_logits": end_logits}
            
        else:  # generative
            logits = torch.tensor(server_response['logits'], device=self.device)
            return {"logits": logits}
    
    def _compute_distillation_loss(
        self,
        local_outputs: Dict[str, torch.Tensor],
        server_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute knowledge distillation loss based on QA format."""
        
        if self.config.qa_format == "multiple_choice":
            local_logits = local_outputs['logits']
            server_logits = server_outputs['logits']
            
            # Temperature-scaled distillation
            temperature = 3.0
            local_probs = torch.log_softmax(local_logits / temperature, dim=-1)
            server_probs = torch.softmax(server_logits / temperature, dim=-1)
            
            return self.distillation_criterion(local_probs, server_probs) * (temperature ** 2)
            
        elif self.config.qa_format == "extractive":
            # Distillation for both start and end logits
            start_loss = self.distillation_criterion(
                torch.log_softmax(local_outputs['start_logits'], dim=-1),
                torch.softmax(server_outputs['start_logits'], dim=-1)
            )
            end_loss = self.distillation_criterion(
                torch.log_softmax(local_outputs['end_logits'], dim=-1),
                torch.softmax(server_outputs['end_logits'], dim=-1)
            )
            
            return (start_loss + end_loss) / 2
            
        else:  # generative
            local_logits = local_outputs['logits']
            server_logits = server_outputs['logits']
            
            # Vocabulary-level distillation
            temperature = 2.0
            local_probs = torch.log_softmax(local_logits / temperature, dim=-1)
            server_probs = torch.softmax(server_logits / temperature, dim=-1)
            
            return self.distillation_criterion(local_probs, server_probs) * (temperature ** 2)
    
    def local_training_epoch(self) -> Dict[str, float]:
        """Perform one epoch of local training with gradient accumulation."""
        
        epoch_metrics = {
            'local_loss': 0.0, 
            'distillation_loss': 0.0, 
            'total_loss': 0.0
        }
        num_steps = 0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Client {self.config.client_id} Training",
            leave=False
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            
            # Training step
            step_metrics = self.local_training_step(batch)
            
            # Update epoch metrics
            for key, value in step_metrics.items():
                epoch_metrics[key] += value
            num_steps += 1
            
            # Gradient accumulation and optimization step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                
                # Gradient clipping
                if self.scaler is not None:
                    # Mixed precision optimization
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.client_model.parameters(), 
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Regular optimization
                    torch.nn.utils.clip_grad_norm_(
                        self.client_model.parameters(), 
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Local Loss': f"{step_metrics['local_loss']:.4f}",
                'Total Loss': f"{step_metrics['total_loss']:.4f}"
            })
        
        # Handle remaining gradients
        if num_steps % self.config.gradient_accumulation_steps != 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.client_model.parameters(), 
                    self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                utils.clip_grad_norm_(
                    self.client_model.parameters(), 
                    self.config.max_grad_norm
                )
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_steps
        
        return epoch_metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the local model on validation data."""
        
        self.client_model.eval()
        eval_metrics = {
            'eval_loss': 0.0,
            'eval_accuracy': 0.0,
            'eval_f1': 0.0
        }
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluation", leave=False):
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.client_model.forward_local(**batch)
                
                # Collect predictions and labels
                predictions, labels = self.metrics.extract_predictions_and_labels(
                    outputs, batch
                )
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                
                # Accumulate loss
                if 'loss' in outputs:
                    eval_metrics['eval_loss'] += outputs['loss'].item()
        
        # Calculate metrics
        eval_metrics['eval_loss'] /= len(self.val_loader)
        
        if all_predictions and all_labels:
            accuracy, f1 = self.metrics.compute_metrics(all_predictions, all_labels)
            eval_metrics['eval_accuracy'] = accuracy
            eval_metrics['eval_f1'] = f1
        
        return eval_metrics
    
    def participate_in_fedavg(self) -> bool:
        """Participate in federated averaging."""
        try:
            # Get current head weights
            head_weights = self.client_model.get_head_weights()
            
            # Convert to serializable format
            serialized_weights = {
                name: tensor.cpu().numpy().tolist() 
                for name, tensor in head_weights.items()
            }
            
            # Send to server
            response = self.communicator.send_weights_for_fedavg(serialized_weights)
            
            if response and 'aggregated_weights' in response:
                # Deserialize and set new weights
                new_weights = {
                    name: torch.tensor(weights, device=self.device)
                    for name, weights in response['aggregated_weights'].items()
                }
                self.client_model.set_head_weights(new_weights)
                
                logger.info("Successfully updated weights from FedAvg")
                return True
            else:
                logger.warning("Failed to receive aggregated weights")
                return False
                
        except Exception as e:
            logger.error(f"Error in FedAvg participation: {e}")
            return False
    
    def train(self, num_rounds: int) -> Dict[str, Any]:
        """Main training loop for the federated QA client."""
        
        training_history = {
            'rounds': [],
            'local_loss': [],
            'distillation_loss': [],
            'total_loss': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'eval_f1': []
        }
        
        for round_num in range(num_rounds):
            logger.info(f"Starting round {round_num + 1}/{num_rounds}")
            
            # Local training for specified epochs
            round_metrics = {'local_loss': 0.0, 'distillation_loss': 0.0, 'total_loss': 0.0}
            
            for epoch in range(self.config.local_epochs):
                epoch_metrics = self.local_training_epoch()
                
                for key, value in epoch_metrics.items():
                    round_metrics[key] += value
            
            # Average over epochs
            for key in round_metrics:
                round_metrics[key] /= self.config.local_epochs
            
            # Evaluation
            eval_metrics = self.evaluate()
            
            # Log metrics
            logger.info(
                f"Round {round_num + 1} - "
                f"Local Loss: {round_metrics['local_loss']:.4f}, "
                f"Eval Loss: {eval_metrics['eval_loss']:.4f}, "
                f"Eval Acc: {eval_metrics['eval_accuracy']:.4f}, "
                f"Eval F1: {eval_metrics['eval_f1']:.4f}"
            )
            
            # Store metrics
            training_history['rounds'].append(round_num + 1)
            training_history['local_loss'].append(round_metrics['local_loss'])
            training_history['distillation_loss'].append(round_metrics['distillation_loss'])
            training_history['total_loss'].append(round_metrics['total_loss'])
            training_history['eval_loss'].append(eval_metrics['eval_loss'])
            training_history['eval_accuracy'].append(eval_metrics['eval_accuracy'])
            training_history['eval_f1'].append(eval_metrics['eval_f1'])
            
            # Participate in FedAvg periodically
            if (round_num + 1) % self.config.fedavg_frequency == 0:
                self.participate_in_fedavg()
        
        return training_history
