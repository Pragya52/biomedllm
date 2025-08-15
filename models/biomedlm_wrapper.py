import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional, Tuple, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

class FullBioMedLMWrapper(nn.Module):
    """Wrapper for the full Stanford BioMedLM model with QA capabilities."""
    
    def __init__(
        self, 
        model_name: str = "stanford-crfm/BioMedLM",
        model_revision: str = "main",
        split_layer: int = 16,
        qa_format: str = "multiple_choice",
        max_answer_length: int = 128,
        cache_dir: str = "./cache"
    ):
        super().__init__()
        
        self.model_name = model_name
        self.model_revision = model_revision
        self.split_layer = split_layer
        self.qa_format = qa_format
        self.max_answer_length = max_answer_length
        
        logger.info(f"Loading full BioMedLM model: {model_name}")
        
        try:
            # Load the full BioMedLM model
            self.config = AutoConfig.from_pretrained(
                model_name,
                revision=model_revision,
                cache_dir=cache_dir
            )
            
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=model_revision,
                config=self.config,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,  # Use half precision for memory efficiency
                device_map="auto"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                revision=model_revision,
                cache_dir=cache_dir
            )
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Successfully loaded full BioMedLM model")
            
        except Exception as e:
            logger.error(f"Failed to load BioMedLM: {e}")
            logger.info("Falling back to GPT-2 for development")
            
            # Fallback to GPT-2 for development/testing
            self.config = AutoConfig.from_pretrained("gpt2")
            self.base_model = AutoModelForCausalLM.from_pretrained("gpt2")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model dimensions
        self.vocab_size = self.config.vocab_size
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        
        # Create QA-specific heads
        self._create_qa_heads()
        
        logger.info(f"Model configuration: {self.num_layers} layers, {self.hidden_size} hidden size")
    
    def _create_qa_heads(self):
        """Create task-specific heads for different QA formats."""
        
        if self.qa_format == "multiple_choice":
            # Classification head for multiple choice
            self.qa_classifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 2, 1)  # Score for each choice
            )
            
        elif self.qa_format == "extractive":
            # Span extraction heads
            self.qa_start_head = nn.Linear(self.hidden_size, 1)
            self.qa_end_head = nn.Linear(self.hidden_size, 1)
            
        elif self.qa_format == "generative":
            # Use the base language modeling head
            self.qa_generator = self.base_model.lm_head
        
        else:
            raise ValueError(f"Unsupported QA format: {self.qa_format}")
    
    def get_embedding_layer(self) -> nn.Module:
        """Get the token embedding layer."""
        if hasattr(self.base_model, 'transformer'):
            return self.base_model.transformer.wte  # GPT-2 style
        elif hasattr(self.base_model, 'model'):
            return self.base_model.model.embed_tokens  # LLaMA style
        else:
            # Generic approach
            for name, module in self.base_model.named_modules():
                if 'embed' in name.lower() and isinstance(module, nn.Embedding):
                    return module
            raise AttributeError("Could not find embedding layer")
    
    def get_transformer_layers(self) -> nn.ModuleList:
        """Get transformer layers."""
        if hasattr(self.base_model, 'transformer'):
            return self.base_model.transformer.h  # GPT-2 style
        elif hasattr(self.base_model, 'model'):
            return self.base_model.model.layers  # LLaMA style
        else:
            raise AttributeError("Could not find transformer layers")
    
    def get_client_components(self) -> Tuple[nn.Module, nn.ModuleList]:
        """Get components for client side (embedding + first layers)."""
        embedding = self.get_embedding_layer()
        transformer_layers = self.get_transformer_layers()
        client_layers = transformer_layers[:self.split_layer]
        return embedding, client_layers
    
    def get_server_components(self) -> Tuple[nn.ModuleList, nn.Module]:
        """Get components for server side (remaining layers + heads)."""
        transformer_layers = self.get_transformer_layers()
        server_layers = transformer_layers[self.split_layer:]
        
        # Get appropriate head based on QA format
        if self.qa_format == "multiple_choice":
            head = self.qa_classifier
        elif self.qa_format == "extractive":
            head = (self.qa_start_head, self.qa_end_head)
        else:  # generative
            head = self.qa_generator
        
        return server_layers, head
    
    def forward_client_side(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for client side (embedding + first layers)."""
        
        # Get embedding
        embedding_layer = self.get_embedding_layer()
        hidden_states = embedding_layer(input_ids)
        
        # Apply positional embeddings if they exist
        if hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'wpe'):
            position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_embeds = self.base_model.transformer.wpe(position_ids)
            hidden_states = hidden_states + position_embeds
        
        # Apply first transformer layers
        transformer_layers = self.get_transformer_layers()
        
        for i in range(self.split_layer):
            layer = transformer_layers[i]
            
            if attention_mask is not None:
                # Create causal mask for GPT-style models
                seq_len = input_ids.size(-1)
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
                causal_mask = causal_mask.bool()
                
                # Combine with attention mask
                combined_mask = attention_mask.unsqueeze(1).unsqueeze(1) * causal_mask
            else:
                combined_mask = None
            
            # Forward through layer
            if hasattr(layer, '__call__'):
                if combined_mask is not None:
                    hidden_states = layer(hidden_states, attention_mask=combined_mask)[0]
                else:
                    hidden_states = layer(hidden_states)[0]
            else:
                hidden_states = layer(hidden_states)
        
        return hidden_states
    
    def forward_server_side(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for server side (remaining layers + QA head)."""
        
        # Apply remaining transformer layers
        transformer_layers = self.get_transformer_layers()
        
        for i in range(self.split_layer, self.num_layers):
            layer = transformer_layers[i]
            
            if attention_mask is not None:
                seq_len = hidden_states.size(1)
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device))
                causal_mask = causal_mask.bool()
                combined_mask = attention_mask.unsqueeze(1).unsqueeze(1) * causal_mask
            else:
                combined_mask = None
            
            # Forward through layer
            if hasattr(layer, '__call__'):
                if combined_mask is not None:
                    hidden_states = layer(hidden_states, attention_mask=combined_mask)[0]
                else:
                    hidden_states = layer(hidden_states)[0]
            else:
                hidden_states = layer(hidden_states)
        
        # Apply final layer norm if it exists
        if hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'ln_f'):
            hidden_states = self.base_model.transformer.ln_f(hidden_states)
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'norm'):
            hidden_states = self.base_model.model.norm(hidden_states)
        
        # Apply QA-specific head
        return self._apply_qa_head(hidden_states, labels, **kwargs)
    
    def _apply_qa_head(
        self, 
        hidden_states: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Apply QA-specific head based on format."""
        
        if self.qa_format == "multiple_choice":
            return self._apply_multiple_choice_head(hidden_states, labels, **kwargs)
        elif self.qa_format == "extractive":
            return self._apply_extractive_head(hidden_states, labels, **kwargs)
        else:  # generative
            return self._apply_generative_head(hidden_states, labels)
    
    def _apply_multiple_choice_head(
        self, 
        hidden_states: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        choice_input_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Apply multiple choice classification head."""
        
        # Pool hidden states (use last token or mean pooling)
        pooled_output = hidden_states.mean(dim=1)  # Mean pooling
        
        # For multiple choice, we need to score each choice
        if choice_input_ids is not None:
            batch_size, num_choices, seq_len = choice_input_ids.shape
            
            # Reshape for processing
            choice_hidden = hidden_states.view(batch_size * num_choices, -1, hidden_states.size(-1))
            choice_pooled = choice_hidden.mean(dim=1)
            
            # Score each choice
            choice_scores = self.qa_classifier(choice_pooled)  # [batch*choices, 1]
            logits = choice_scores.view(batch_size, num_choices)  # [batch, choices]
        else:
            # Single sequence classification
            logits = self.qa_classifier(pooled_output)
        
        outputs = {"logits": logits}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            outputs["loss"] = loss
        
        return outputs
    
    def _apply_extractive_head(
        self, 
        hidden_states: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Apply extractive QA heads for span extraction."""
        
        # Get start and end logits
        start_logits = self.qa_start_head(hidden_states).squeeze(-1)
        end_logits = self.qa_end_head(hidden_states).squeeze(-1)
        
        outputs = {
            "start_logits": start_logits,
            "end_logits": end_logits
        }
        
        if start_positions is not None and end_positions is not None:
            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs["loss"] = total_loss
        
        return outputs
    
    def _apply_generative_head(
        self, 
        hidden_states: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Apply generative language modeling head."""
        
        # Apply language modeling head
        logits = self.qa_generator(hidden_states)
        
        outputs = {"logits": logits}
        
        if labels is not None:
            # Shift labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs["loss"] = loss
        
        return outputs
    
    def generate_answer(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """Generate answer for generative QA."""
        
        if self.qa_format != "generative":
            raise ValueError("Answer generation only available for generative QA format")
        
        with torch.no_grad():
            # Generate using the full model
            generated = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            
            # Extract answer part (after "Answer:")
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text.strip()
            
            return answer# Federated Learning with BioMedLM - Real Medical QA Implementation
