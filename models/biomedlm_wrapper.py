"""
BiomedLLM Wrapper using GPT4All from ModelScope
Updated implementation replacing Stanford BioMedLM with GPT4All model
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from modelscope import snapshot_download
from typing import Optional, Tuple, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class FullBioMedLMWrapper(nn.Module):
    """Wrapper for the GPT4All model with QA capabilities from ModelScope."""
    
    def __init__(
        self, 
        model_name: str = 'Genius-Society/gpt4all',
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
        self._is_loaded = False
        
        logger.info(f"Loading GPT4All model from ModelScope: {model_name}")
        
        try:
            # First try to download from ModelScope
            logger.info(f"Attempting to download GPT4All model from ModelScope: {model_name}")
            self.model_dir = snapshot_download(
                self.model_name,
                revision=model_revision,
                cache_dir=cache_dir
            )
            
            logger.info(f"Model downloaded to: {self.model_dir}")
            
            # Load config first
            self.config = AutoConfig.from_pretrained(
                self.model_dir,
                trust_remote_code=True
            )
            
            # Load the GPT4All model
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                config=self.config,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=False
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                trust_remote_code=True
            )
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Successfully loaded GPT4All model from ModelScope")
            
        except Exception as e:
            logger.error(f"Failed to load GPT4All from ModelScope: {e}")
            logger.info("Falling back to Hugging Face Hub...")
            
            try:
                # Fallback: Load directly from Hugging Face
                logger.info("Loading GPT4All model directly from Hugging Face...")
                
                self.config = AutoConfig.from_pretrained(
                    self.model_name,
                    revision=model_revision,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
                
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    revision=model_revision,
                    config=self.config,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float32,
                    device_map=None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=False
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    revision=model_revision,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
                
                # Ensure pad token exists
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Set model_dir for consistency
                self.model_dir = self.model_name
                
                logger.info("Successfully loaded GPT4All model from Hugging Face")
                
            except Exception as e2:
                logger.error(f"Failed to load GPT4All from Hugging Face: {e2}")
                
                try:
                    logger.info("Trying alternative GPT4All loading method with float16...")
                    
                    self.base_model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        revision=model_revision,
                        cache_dir=cache_dir,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        device_map=None,
                        trust_remote_code=True
                    )
                    
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.base_model = self.base_model.to(device)
                    
                    self.config = self.base_model.config
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        revision=model_revision,
                        cache_dir=cache_dir,
                        trust_remote_code=True
                    )
                    
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    # Set model_dir for consistency
                    self.model_dir = self.model_name
                    
                    logger.info("Successfully loaded GPT4All with alternative method")
                    
                except Exception as e3:
                    logger.error(f"Failed to load GPT4All with alternative method: {e3}")
                    raise RuntimeError("Could not load GPT4All model. Please check the model name and ensure it's available on ModelScope or Hugging Face.")
        
        # Model dimensions
        self.vocab_size = self.config.vocab_size
        self.hidden_size = getattr(self.config, 'hidden_size', getattr(self.config, 'n_embd', 768))
        self.num_layers = getattr(self.config, 'num_hidden_layers', getattr(self.config, 'n_layer', 12))
        
        # Create QA-specific heads
        self._create_qa_heads()
        
        logger.info(f"GPT4All configuration: {self.num_layers} layers, {self.hidden_size} hidden size")

    
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
            return self.base_model.transformer.wte  # GPT-2/GPT4All style
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
            return self.base_model.transformer.h  # GPT-2/GPT4All style
        elif hasattr(self.base_model, 'model'):
            return self.base_model.model.layers  # LLaMA style
        else:
            raise AttributeError("Could not find transformer layers")
    
    def get_client_components(self) -> Tuple[nn.Module, nn.ModuleList]:
        """Get components for client side (embedding + first layers)."""
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Please call ensure_loaded() first.")
        
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
            
            return answer
    
    def ensure_loaded(self, device):
        """Ensure model is properly loaded and not using meta tensors."""
        if self._is_loaded:
            return
            
        try:
            # If model was created with meta device, properly load it
            if hasattr(self.base_model, 'parameters') and any(p.is_meta for p in self.base_model.parameters()):
                logger.info("Converting meta tensors to actual tensors...")
                self.base_model = self.base_model.to_empty(device=device)
                
                # Load state dict if available
                if hasattr(self, 'checkpoint_path') and self.checkpoint_path:
                    state_dict = torch.load(self.checkpoint_path, map_location=device)
                    self.base_model.load_state_dict(state_dict)
                else:
                    # Initialize with random weights if no checkpoint
                    for param in self.base_model.parameters():
                        if param.requires_grad:
                            torch.nn.init.normal_(param, mean=0, std=0.02)
            
            self._is_loaded = True
            logger.info("GPT4All model successfully loaded and ready for use")
            
        except Exception as e:
            logger.error(f"Failed to ensure GPT4All model is loaded: {e}")
            raise
    
    # Additional utility methods for GPT4All integration
    def encode_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Encode text into tokens using GPT4All tokenizer."""
        return self.tokenizer.encode(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
    
    def decode_tokens(self, tokens: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode tokens back to text using GPT4All tokenizer."""
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """Generate text using GPT4All model."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from the generated text
        generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded GPT4All model."""
        return {
            "model_name": self.model_name,
            "model_dir": getattr(self, 'model_dir', 'Not available'),
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "qa_format": self.qa_format,
            "split_layer": self.split_layer,
            "max_answer_length": self.max_answer_length
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize the wrapper with GPT4All
    wrapper = FullBioMedLMWrapper(
        model_name='Genius-Society/gpt4all',
        qa_format="generative",
        split_layer=8,
        cache_dir="./models"
    )
    
    # Test text generation
    prompt = "The symptoms of diabetes include"
    generated_text = wrapper.generate_text(
        prompt,
        max_new_tokens=100,
        temperature=0.8
    )
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    # Test QA functionality
    qa_prompt = "Question: What are the main symptoms of diabetes? Answer:"
    inputs = wrapper.tokenizer(qa_prompt, return_tensors="pt")
    
    answer = wrapper.generate_answer(
        input_ids=inputs['input_ids'],
        attention_mask=inputs.get('attention_mask'),
        max_new_tokens=128
    )
    
    print(f"\nQA Prompt: {qa_prompt}")
    print(f"Answer: {answer}")
    
    # Get model info
    model_info = wrapper.get_model_info()
    print(f"\nModel Info: {model_info}")
    
    # Test federated learning components
    try:
        client_embedding, client_layers = wrapper.get_client_components()
        server_layers, server_head = wrapper.get_server_components()
        print(f"\nFederated Learning Setup:")
        print(f"Client layers: {len(client_layers)}")
        print(f"Server layers: {len(server_layers)}")
        print(f"QA format: {wrapper.qa_format}")
    except Exception as e:
        print(f"Federated learning setup error: {e}")
        print("Make sure to call ensure_loaded() first if needed")
