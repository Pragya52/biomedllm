import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from models.biomedlm_wrapper import FullBioMedLMWrapper
import logging

logger = logging.getLogger(__name__)

class ClientQAModel(nn.Module):
    """Client-side model for federated medical QA learning."""
    
    def __init__(
        self,
        biomedlm_wrapper: FullBioMedLMWrapper,
        local_layers: int = 4,
        client_id: str = "client_1",
        device: torch.device=None
    ):
        super().__init__()
        
        self.client_id = client_id
        self.local_layers = local_layers
        self.device=device or torch.device("cpu")

        if not hasattr(biomedlm_wrapper, "_is_loaded") or not biomedlm_wrapper._is_loaded:
                       biomedlm_wrapper.ensure_loaded(self.device)


        self.biomedlm_wrapper = biomedlm_wrapper
        self.hidden_size = biomedlm_wrapper.hidden_size
        self.qa_format = biomedlm_wrapper.qa_format

        try:
        
        # Get client components from BioMedLM
            self.embedding, self.transformer_layers = biomedlm_wrapper.get_client_components()

            if hasattr(self.embedding, 'weight') and self.embedding.weight.is_meta:
                self.embedding = self.embedding.to_empty(device=self.device)

            for i,layer in enumerate(self.transformer_layers):
                if any(p.is_meta for p in layer.parameters()):
                    self.transformer_layers[i] = layer.to_empty(device=self.device)
        except Exception as e:
            logger.error(f"Error getting client componenets: {e}")
            raise RuntimeError("Failed to initialize client model components:{e}")

        
        # Local processing layers for knowledge distillation
        self.local_processor = nn.ModuleList([
            self._create_local_layer() for _ in range(local_layers)
        ])
        
        # Local QA head for distillation
        self.local_qa_head = self._create_local_qa_head()

        self.local_processor=self.local_processor.to(self.device)
        self.local_qa_head=self.local_qa_head.to(self.device)
        
        logger.info(f"Initialized client model {client_id} with {local_layers} local layers")
    
    def _create_local_layer(self) -> nn.Module:
        """Create a lightweight local processing layer."""
        return nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=8,  # Reduced from full model
            dim_feedforward=self.hidden_size * 2,  # Reduced feedforward
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
    
    def _create_local_qa_head(self) -> nn.Module:
        """Create local QA head based on format."""
        if self.qa_format == "multiple_choice":
            return nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 4),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 4, 4)  # 4 choices typically
            )
        elif self.qa_format == "extractive":
            return nn.ModuleDict({
                'start_head': nn.Linear(self.hidden_size, 1),
                'end_head': nn.Linear(self.hidden_size, 1)
            })
        else:  # generative
            return nn.Linear(self.hidden_size, self.biomedlm_wrapper.vocab_size)

    def forward_local(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Local forward pass for knowledge distillation."""
        
        # Get embeddings
        hidden_states = self.embedding(input_ids)
        
        # Apply local processing layers
        for layer in self.local_processor:
            if attention_mask is not None:
                # Create attention mask for transformer layer
                #extended_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                #extended_mask = extended_mask.expand(-1, -1, attention_mask.size(-1), -1)
                #extended_mask = (1.0 - extended_mask) * -10000.0
                key_padding_mask=(attention_mask==0)
            else:
                key_padding_mask = None
            
            hidden_states = layer(hidden_states, src_key_padding_mask=key_padding_mask)
        
        # Apply local QA head
        return self._apply_local_qa_head(hidden_states, **kwargs)
    
    def forward_global_path(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for global path (to be sent to server)."""
        return self.biomedlm_wrapper.forward_client_side(input_ids, attention_mask)
    
    def _apply_local_qa_head(
        self, 
        hidden_states: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Apply local QA head based on format."""
        
        if self.qa_format == "multiple_choice":
            # Pool and classify
            pooled_output = hidden_states.mean(dim=1)
            logits = self.local_qa_head(pooled_output)
            
            outputs = {"logits": logits}
            
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                outputs["loss"] = loss
            
            return outputs
            
        elif self.qa_format == "extractive":
            # Extract spans
            start_logits = self.local_qa_head['start_head'](hidden_states).squeeze(-1)
            end_logits = self.local_qa_head['end_head'](hidden_states).squeeze(-1)
            
            outputs = {
                "start_logits": start_logits,
                "end_logits": end_logits
            }
            
            if 'start_positions' in kwargs and 'end_positions' in kwargs:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(start_logits, kwargs['start_positions'])
                end_loss = loss_fct(end_logits, kwargs['end_positions'])
                outputs["loss"] = (start_loss + end_loss) / 2
            
            return outputs
            
        else:  # generative
            logits = self.local_qa_head(hidden_states)
            outputs = {"logits": logits}
            
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                outputs["loss"] = loss
            
            return outputs
    
    def get_head_weights(self) -> Dict[str, torch.Tensor]:
        """Get embedding weights for FedAvg."""
        return {
            name: param.clone().detach() 
            for name, param in self.embedding.named_parameters()
        }
    
    def set_head_weights(self, weights: Dict[str, torch.Tensor]):
        """Set embedding weights from FedAvg."""
        with torch.no_grad():
            for name, param in self.embedding.named_parameters():
                if name in weights:
                    param.copy_(weights[name])

class ServerQAModel(nn.Module):
    """Server-side model for federated medical QA learning."""
    
    def __init__(self, biomedlm_wrapper: FullBioMedLMWrapper):
        super().__init__()
        
        self.biomedlm_wrapper = biomedlm_wrapper
        self.hidden_size = biomedlm_wrapper.hidden_size
        self.qa_format = biomedlm_wrapper.qa_format
        
        # Get server components from BioMedLM
        self.transformer_layers, self.qa_heads = biomedlm_wrapper.get_server_components()
        
        logger.info(f"Initialized server model for {self.qa_format} QA")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Server forward pass."""
        return self.biomedlm_wrapper.forward_server_side(
            hidden_states, attention_mask, labels, **kwargs
        )
