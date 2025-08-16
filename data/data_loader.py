import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Any, Optional, Tuple
import logging
from typing import Dict, List, Any, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class MedicalQADataset(Dataset):
    """Dataset for medical question-answering tasks."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        max_answer_length: int = 128,
        qa_format: str = "extractive"
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_answer_length = max_answer_length
        self.qa_format = qa_format
        
        # Ensure tokenizer has required tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        if self.qa_format == "extractive":
            return self._process_extractive_qa(example)
        elif self.qa_format == "multiple_choice":
            return self._process_multiple_choice_qa(example)
        elif self.qa_format == "generative":
            return self._process_generative_qa(example)
        else:
            raise ValueError(f"Unsupported QA format: {self.qa_format}")

    def _process_extractive_qa(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process extractive QA (answer span extraction from context)."""
        question = example["question"]
        context = example.get("context", "")
        answer = example["answer"]
        
        # Create input text
        if context:
            input_text = f"Question: {question}\nContext: {context}"
        else:
            input_text = f"Question: {question}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize answer for span extraction
        answer_tokens = self.tokenizer(
            answer,
            truncation=True,
            max_length=self.max_answer_length,
            return_tensors="pt"
        )
        
        # Find answer span in context (simplified)
        start_pos = 0
        end_pos = 0
        if context and answer.lower() in context.lower():
            context_lower = context.lower()
            answer_lower = answer.lower()
            start_char = context_lower.find(answer_lower)
            if start_char != -1:
                end_char = start_char + len(answer_lower)
                
                # Convert character positions to token positions
                context_start = input_text.find(context)
                if context_start != -1:
                    abs_start = context_start + start_char
                    abs_end = context_start + end_char
                    
                    # Encode and find token positions
                    encoded = self.tokenizer(input_text[:abs_start], return_tensors="pt")
                    start_pos = len(encoded["input_ids"][0]) - 1
                    
                    encoded = self.tokenizer(input_text[:abs_end], return_tensors="pt")
                    end_pos = len(encoded["input_ids"][0]) - 1
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "start_positions": torch.tensor(start_pos, dtype=torch.long),
            "end_positions": torch.tensor(end_pos, dtype=torch.long),
            "answer_text": answer,
            "question_type": example.get("question_type", "extractive")
        }
    
    def _process_multiple_choice_qa(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process multiple choice QA."""
        question = example["question"]
        choices = example.get("choices", [])
        answer = example["answer"]
        context = example.get("context", "")
        
        # Find correct choice index
        correct_choice_idx = 0
        if answer in choices:
            correct_choice_idx = choices.index(answer)
        
        # Format input with context if available
        if context:
            input_text = f"Context: {context}\n\nQuestion: {question}"
        else:
            input_text = f"Question: {question}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize each choice for classification
        choice_inputs = []
        for choice in choices:
            choice_text = f"{input_text}\nAnswer: {choice}"
            choice_encoding = self.tokenizer(
                choice_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            choice_inputs.append(choice_encoding)
        
        # Pad choices to fixed number (4 for standard multiple choice)
        max_choices = 4
        while len(choice_inputs) < max_choices:
            # Add dummy choice
            dummy_encoding = self.tokenizer(
                f"{input_text}\nAnswer: [DUMMY]",
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            choice_inputs.append(dummy_encoding)
        
        # Stack choice inputs
        choice_input_ids = torch.stack([c["input_ids"].squeeze() for c in choice_inputs])
        choice_attention_masks = torch.stack([c["attention_mask"].squeeze() for c in choice_inputs])
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "choice_input_ids": choice_input_ids,
            "choice_attention_masks": choice_attention_masks,
            "labels": torch.tensor(correct_choice_idx, dtype=torch.long),
            "num_choices": len(choices),
            "question_type": example.get("question_type", "multiple_choice")
        }
    
    def _process_generative_qa(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process generative QA (generate answer text)."""
        question = example["question"]
        answer = example["answer"]
        context = example.get("context", "")
        
        # Format input
        if context:
            input_text = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        else:
            input_text = f"Question: {question}\nAnswer:"
        
        # Format target with answer
        target_text = f"{input_text} {answer}"
        
        # Tokenize input and target
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        targets = self.tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create labels (shift target tokens for language modeling)
        labels = targets["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
            "target_text": answer,
            "question_type": example.get("question_type", "generative")
        }

class FederatedQADataLoader:
    """Data loader for federated medical QA learning."""
    
    def __init__(self, client_id: str, config: Any):
        self.client_id = client_id
        self.config = config
        
        # Initialize tokenizer for full BioMedLM
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            revision=config.model_revision,
            cache_dir="./cache"
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info(f"Initialized tokenizer for {config.model_name}")
        
    def load_client_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load training and validation data for this client."""
        
        # Load client-specific data
        train_data = self._load_data_split("train")
        val_data = self._load_data_split("val")
        
        # Create datasets
        train_dataset = MedicalQADataset(
            data=train_data,
            tokenizer=self.tokenizer,
            max_length=self.config.max_sequence_length,
            max_answer_length=self.config.max_answer_length,
            qa_format=self.config.qa_format
        )
        
        val_dataset = MedicalQADataset(
            data=val_data,
            tokenizer=self.tokenizer,
            max_length=self.config.max_sequence_length,
            max_answer_length=self.config.max_answer_length,
            qa_format=self.config.qa_format
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.config.device == "cuda" else False,
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.config.device == "cuda" else False,
            collate_fn=self._collate_fn
        )
        
        logger.info(f"Loaded data for {self.client_id}: {len(train_dataset)} train, {len(val_dataset)} val")
        
        return train_loader, val_loader
    
    def _load_data_split(self, split: str) -> List[Dict[str, Any]]:
        """Load data for a specific split (train/val)."""
        data_path = f"./data/{self.client_id}/{split}.json"
        
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            logger.warning(f"Data file not found: {data_path}. Using dummy data.")
            return self._create_dummy_data(split)
    
    def _create_dummy_data(self, split: str) -> List[Dict[str, Any]]:
        """Create dummy medical QA data for testing."""
        dummy_data = [
            {
                "question": "What is the most common cause of chest pain in emergency departments?",
                "context": "Chest pain is a common presenting complaint in emergency medicine.",
                "answer": "Myocardial infarction",
                "question_type": self.config.qa_format,
                "choices": ["Myocardial infarction", "Pneumonia", "GERD", "Anxiety"] if self.config.qa_format == "multiple_choice" else [],
                "metadata": {"id": f"dummy_{split}_1", "source": "dummy"}
            },
            {
                "question": "Which blood test is most specific for cardiac muscle damage?",
                "context": "Cardiac biomarkers are proteins released when heart muscle is damaged.",
                "answer": "Troponin",
                "question_type": self.config.qa_format,
                "choices": ["Troponin", "CK-MB", "Myoglobin", "LDH"] if self.config.qa_format == "multiple_choice" else [],
                "metadata": {"id": f"dummy_{split}_2", "source": "dummy"}
            }
        ]
        
        # Repeat data to simulate larger dataset
        num_samples = 100 if split == "train" else 20
        return (dummy_data * (num_samples // len(dummy_data) + 1))[:num_samples]
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        if not batch:
            return {}
        
        # Determine QA format from first item
        qa_format = batch[0].get("question_type", self.config.qa_format)
        
        if qa_format == "multiple_choice":
            return self._collate_multiple_choice(batch)
        elif qa_format == "extractive":
            return self._collate_extractive(batch)
        else:  # generative
            return self._collate_generative(batch)
    
    def _collate_multiple_choice(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate multiple choice QA batch."""
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "choice_input_ids": torch.stack([item["choice_input_ids"] for item in batch]),
            "choice_attention_masks": torch.stack([item["choice_attention_masks"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
            "question_type": batch[0]["question_type"]
        }
    
    def _collate_extractive(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate extractive QA batch."""
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "start_positions": torch.stack([item["start_positions"] for item in batch]),
            "end_positions": torch.stack([item["end_positions"] for item in batch]),
            "question_type": batch[0]["question_type"]
        }
    
    def _collate_generative(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate generative QA batch."""
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
            "question_type": batch[0]["question_type"]
        }
