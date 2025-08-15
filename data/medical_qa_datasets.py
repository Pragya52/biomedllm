
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datasets import load_dataset, Dataset, DatasetDict
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class MedicalQADatasetManager:
    """Manager for real medical QA datasets."""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Available medical QA datasets
        self.available_datasets = {
            "medqa": {
                "hf_name": "bigbio/med_qa",
                "config": "med_qa_en_bigbio_qa",
                "description": "Medical board exam questions with multiple choice answers"
            },
            "pubmedqa": {
                "hf_name": "pubmed_qa",
                "config": "pqa_labeled",
                "description": "PubMed abstract based QA with yes/no/maybe answers"
            },
            "bioasq": {
                "hf_name": "bioasq",
                "config": "8b",
                "description": "Biomedical question answering from literature"
            },
            "mmlu_medicine": {
                "hf_name": "cais/mmlu",
                "config": "anatomy",
                "description": "Medical subset of MMLU benchmark"
            },
            "medmcqa": {
                "hf_name": "medmcqa", 
                "config": None,
                "description": "Indian medical entrance exam questions"
            }
        }
    
    def load_dataset(self, dataset_name: str, split: str = "train") -> Dataset:
        """Load a specific medical QA dataset."""
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Dataset {dataset_name} not available. Choose from: {list(self.available_datasets.keys())}")
        
        dataset_info = self.available_datasets[dataset_name]
        
        try:
            logger.info(f"Loading {dataset_name} dataset...")
            
            if dataset_name == "medqa":
                return self._load_medqa(split)
            elif dataset_name == "pubmedqa":
                return self._load_pubmedqa(split)
            elif dataset_name == "bioasq":
                return self._load_bioasq(split)
            elif dataset_name == "mmlu_medicine":
                return self._load_mmlu_medicine(split)
            elif dataset_name == "medmcqa":
                return self._load_medmcqa(split)
            else:
                # Generic loading
                dataset = load_dataset(
                    dataset_info["hf_name"],
                    dataset_info["config"],
                    split=split,
                    cache_dir=self.cache_dir
                )
                return self._standardize_format(dataset, dataset_name)
                
        except Exception as e:
            logger.error(f"Error loading {dataset_name}: {e}")
            logger.info("Falling back to synthetic medical QA data...")
            return self._create_synthetic_medical_qa(split)
    
    def _load_medqa(self, split: str) -> Dataset:
        """Load MedQA dataset with standardized format."""
        try:
            dataset = load_dataset(
                "bigbio/med_qa",
                "med_qa_en_bigbio_qa",
                split=split,
                cache_dir=self.cache_dir
            )
            
            # Standardize MedQA format
            def format_medqa(example):
                question = example["question"]
                choices = example.get("choices", [])
                answer = example.get("answer", [""])[0] if example.get("answer") else ""
                
                # Format multiple choice
                if choices:
                    choice_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                    formatted_question = f"{question}\n\nChoices:\n{choice_text}"
                else:
                    formatted_question = question
                
                return {
                    "question": formatted_question,
                    "context": example.get("context", ""),
                    "answer": answer,
                    "question_type": "multiple_choice",
                    "choices": choices,
                    "metadata": {
                        "id": example.get("id", ""),
                        "source": "medqa"
                    }
                }
            
            return dataset.map(format_medqa)
            
        except Exception as e:
            logger.warning(f"Could not load MedQA: {e}")
            return self._create_synthetic_medical_qa(split, "multiple_choice")
    
    def _load_pubmedqa(self, split: str) -> Dataset:
        """Load PubMedQA dataset with standardized format."""
        try:
            dataset = load_dataset(
                "pubmed_qa",
                "pqa_labeled",
                split=split,
                cache_dir=self.cache_dir
            )
            
            def format_pubmedqa(example):
                question = example["question"]
                context = example.get("context", {})
                contexts = context.get("contexts", []) if isinstance(context, dict) else []
                context_text = " ".join(contexts) if contexts else ""
                
                # Map yes/no/maybe to standardized format
                answer_map = {"yes": "Yes", "no": "No", "maybe": "Maybe"}
                answer = answer_map.get(example.get("final_decision", "").lower(), "Unknown")
                
                return {
                    "question": question,
                    "context": context_text,
                    "answer": answer,
                    "question_type": "yes_no_maybe",
                    "choices": ["Yes", "No", "Maybe"],
                    "metadata": {
                        "id": example.get("pubid", ""),
                        "source": "pubmedqa"
                    }
                }
            
            return dataset.map(format_pubmedqa)
            
        except Exception as e:
            logger.warning(f"Could not load PubMedQA: {e}")
            return self._create_synthetic_medical_qa(split, "yes_no_maybe")
    
    def _load_bioasq(self, split: str) -> Dataset:
        """Load BioASQ dataset with standardized format."""
        try:
            # BioASQ requires special handling - using synthetic for now
            logger.info("BioASQ requires special access - using synthetic biomedical QA")
            return self._create_synthetic_medical_qa(split, "extractive")
            
        except Exception as e:
            logger.warning(f"Could not load BioASQ: {e}")
            return self._create_synthetic_medical_qa(split, "extractive")
    
    def _load_mmlu_medicine(self, split: str) -> Dataset:
        """Load medical subsets of MMLU."""
        try:
            medical_subjects = ["anatomy", "clinical_knowledge", "medical_genetics", 
                              "professional_medicine", "college_medicine"]
            
            all_data = []
            for subject in medical_subjects:
                try:
                    dataset = load_dataset(
                        "cais/mmlu",
                        subject,
                        split=split,
                        cache_dir=self.cache_dir
                    )
                    
                    for example in dataset:
                        choices = example.get("choices", [])
                        answer_idx = example.get("answer", 0)
                        answer = choices[answer_idx] if answer_idx < len(choices) else ""
                        
                        choice_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                        formatted_question = f"{example['question']}\n\nChoices:\n{choice_text}"
                        
                        all_data.append({
                            "question": formatted_question,
                            "context": "",
                            "answer": answer,
                            "question_type": "multiple_choice",
                            "choices": choices,
                            "metadata": {
                                "id": f"{subject}_{len(all_data)}",
                                "source": f"mmlu_{subject}"
                            }
                        })
                        
                except Exception as e:
                    logger.warning(f"Could not load MMLU {subject}: {e}")
                    continue
            
            if all_data:
                return Dataset.from_list(all_data)
            else:
                return self._create_synthetic_medical_qa(split, "multiple_choice")
                
        except Exception as e:
            logger.warning(f"Could not load MMLU medicine: {e}")
            return self._create_synthetic_medical_qa(split, "multiple_choice")
    
    def _load_medmcqa(self, split: str) -> Dataset:
        """Load MedMCQA dataset."""
        try:
            dataset = load_dataset(
                "medmcqa",
                split=split,
                cache_dir=self.cache_dir
            )
            
            def format_medmcqa(example):
                question = example["question"]
                choices = [example.get("opa", ""), example.get("opb", ""), 
                          example.get("opc", ""), example.get("opd", "")]
                choices = [c for c in choices if c]  # Remove empty choices
                
                answer_idx = example.get("cop", 0)
                answer = choices[answer_idx] if answer_idx < len(choices) else ""
                
                choice_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                formatted_question = f"{question}\n\nChoices:\n{choice_text}"
                
                return {
                    "question": formatted_question,
                    "context": example.get("exp", ""),  # Explanation as context
                    "answer": answer,
                    "question_type": "multiple_choice",
                    "choices": choices,
                    "metadata": {
                        "id": example.get("id", ""),
                        "source": "medmcqa",
                        "subject": example.get("subject_name", "")
                    }
                }
            
            return dataset.map(format_medmcqa)
            
        except Exception as e:
            logger.warning(f"Could not load MedMCQA: {e}")
            return self._create_synthetic_medical_qa(split, "multiple_choice")
    
    def _create_synthetic_medical_qa(self, split: str, qa_type: str = "multiple_choice") -> Dataset:
        """Create synthetic medical QA data as fallback."""
        logger.info(f"Creating synthetic medical QA data for {split} split")
        
        # Medical QA templates
        templates = {
            "multiple_choice": [
                {
                    "question": "A 45-year-old patient presents with chest pain and shortness of breath. What is the most likely diagnosis?",
                    "choices": ["Myocardial infarction", "Pneumonia", "Gastroesophageal reflux", "Anxiety disorder"],
                    "answer": "Myocardial infarction",
                    "context": "Patient has risk factors including hypertension and smoking history."
                },
                {
                    "question": "Which medication is first-line treatment for hypertension in elderly patients?",
                    "choices": ["ACE inhibitors", "Beta blockers", "Calcium channel blockers", "Diuretics"],
                    "answer": "ACE inhibitors", 
                    "context": "Current guidelines recommend ACE inhibitors for initial therapy."
                },
                {
                    "question": "What is the normal range for HbA1c in diabetic patients?",
                    "choices": ["<6%", "<7%", "<8%", "<9%"],
                    "answer": "<7%",
                    "context": "American Diabetes Association guidelines recommend HbA1c <7% for most adults."
                }
            ],
            "yes_no_maybe": [
                {
                    "question": "Does aspirin reduce the risk of cardiovascular events?",
                    "choices": ["Yes", "No", "Maybe"],
                    "answer": "Yes",
                    "context": "Multiple studies show aspirin reduces cardiovascular risk in high-risk patients."
                },
                {
                    "question": "Is antibiotic therapy effective for viral infections?",
                    "choices": ["Yes", "No", "Maybe"], 
                    "answer": "No",
                    "context": "Antibiotics are only effective against bacterial infections, not viral."
                }
            ],
            "extractive": [
                {
                    "question": "What causes Type 1 diabetes?",
                    "answer": "autoimmune destruction of pancreatic beta cells",
                    "context": "Type 1 diabetes is caused by autoimmune destruction of pancreatic beta cells that produce insulin."
                },
                {
                    "question": "How is tuberculosis transmitted?",
                    "answer": "airborne droplets",
                    "context": "Tuberculosis is transmitted through airborne droplets when infected individuals cough or sneeze."
                }
            ]
        }
        
        # Generate synthetic data
        base_templates = templates.get(qa_type, templates["multiple_choice"])
        num_samples = 1000 if split == "train" else 200
        
        data = []
        for i in range(num_samples):
            template = base_templates[i % len(base_templates)]
            
            # Add variation to avoid overfitting
            question = template["question"]
            if i > 0:
                question = question.replace("45-year-old", f"{40 + (i % 20)}-year-old")
            
            data.append({
                "question": question,
                "context": template["context"],
                "answer": template["answer"],
                "question_type": qa_type,
                "choices": template.get("choices", []),
                "metadata": {
                    "id": f"synthetic_{i}",
                    "source": "synthetic"
                }
            })
        
        return Dataset.from_list(data)
    
    def create_federated_splits(
        self, 
        dataset: Dataset, 
        num_clients: int = 3,
        split_strategy: str = "specialty_based"
    ) -> Dict[int, Dict[str, Dataset]]:
        """Create federated splits for medical QA data."""
        
        if split_strategy == "specialty_based":
            return self._create_specialty_based_splits(dataset, num_clients)
        elif split_strategy == "question_type":
            return self._create_question_type_splits(dataset, num_clients)
        else:  # iid
            return self._create_iid_splits(dataset, num_clients)
    
    def _create_specialty_based_splits(self, dataset: Dataset, num_clients: int) -> Dict[int, Dict[str, Dataset]]:
        """Split data based on medical specialties."""
        # Group by source or inferred specialty
        specialty_groups = {}
        
        for example in dataset:
            source = example["metadata"]["source"]
            specialty = self._infer_specialty(example["question"])
            key = f"{source}_{specialty}"
            
            if key not in specialty_groups:
                specialty_groups[key] = []
            specialty_groups[key].append(example)
        
        # Distribute specialties across clients
        specialty_keys = list(specialty_groups.keys())
        client_splits = {i: {"train": [], "val": []} for i in range(num_clients)}
        
        for i, specialty in enumerate(specialty_keys):
            client_id = i % num_clients
            examples = specialty_groups[specialty]
            
            # Split into train/val
            train_examples, val_examples = train_test_split(
                examples, test_size=0.2, random_state=42
            )
            
            client_splits[client_id]["train"].extend(train_examples)
            client_splits[client_id]["val"].extend(val_examples)
        
        # Convert to datasets
        for client_id in client_splits:
            client_splits[client_id]["train"] = Dataset.from_list(client_splits[client_id]["train"])
            client_splits[client_id]["val"] = Dataset.from_list(client_splits[client_id]["val"])
        
        return client_splits
    
    def _create_question_type_splits(self, dataset: Dataset, num_clients: int) -> Dict[int, Dict[str, Dataset]]:
        """Split data based on question types."""
        type_groups = {"multiple_choice": [], "yes_no_maybe": [], "extractive": []}
        
        for example in dataset:
            q_type = example.get("question_type", "multiple_choice")
            if q_type in type_groups:
                type_groups[q_type].append(example)
        
        # Distribute types across clients
        client_splits = {i: {"train": [], "val": []} for i in range(num_clients)}
        
        for i, (q_type, examples) in enumerate(type_groups.items()):
            if not examples:
                continue
                
            client_id = i % num_clients
            train_examples, val_examples = train_test_split(
                examples, test_size=0.2, random_state=42
            )
            
            client_splits[client_id]["train"].extend(train_examples)
            client_splits[client_id]["val"].extend(val_examples)
        
        # Convert to datasets
        for client_id in client_splits:
            client_splits[client_id]["train"] = Dataset.from_list(client_splits[client_id]["train"])
            client_splits[client_id]["val"] = Dataset.from_list(client_splits[client_id]["val"])
        
        return client_splits
    
    def _create_iid_splits(self, dataset: Dataset, num_clients: int) -> Dict[int, Dict[str, Dataset]]:
        """Create IID splits across clients."""
        # Shuffle and split evenly
        shuffled_indices = np.random.permutation(len(dataset))
        samples_per_client = len(dataset) // num_clients
        
        client_splits = {}
        for client_id in range(num_clients):
            start_idx = client_id * samples_per_client
            end_idx = start_idx + samples_per_client
            
            client_indices = shuffled_indices[start_idx:end_idx]
            client_data = dataset.select(client_indices)
            
            # Split into train/val
            train_size = int(0.8 * len(client_data))
            train_data = client_data.select(range(train_size))
            val_data = client_data.select(range(train_size, len(client_data)))
            
            client_splits[client_id] = {
                "train": train_data,
                "val": val_data
            }
        
        return client_splits
    
    def _infer_specialty(self, question: str) -> str:
        """Infer medical specialty from question content."""
        specialty_keywords = {
            "cardiology": ["heart", "cardiac", "cardiovascular", "chest pain", "hypertension"],
            "endocrinology": ["diabetes", "hormone", "thyroid", "insulin", "glucose"],
            "pulmonology": ["lung", "respiratory", "breathing", "pneumonia", "asthma"],
            "neurology": ["brain", "neurological", "seizure", "stroke", "headache"],
            "gastroenterology": ["stomach", "digestive", "liver", "intestinal", "gastric"],
            "oncology": ["cancer", "tumor", "chemotherapy", "malignant", "metastasis"],
            "infectious_disease": ["infection", "antibiotic", "virus", "bacteria", "fever"],
            "emergency": ["emergency", "trauma", "acute", "urgent", "critical"]
        }
        
        question_lower = question.lower()
        for specialty, keywords in specialty_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return specialty
        
        return "general"
    
    def save_federated_data(
        self, 
        splits: Dict[int, Dict[str, Dataset]], 
        base_path: str = "./data"
    ):
        """Save federated splits to disk."""
        for client_id, data_splits in splits.items():
            client_dir = os.path.join(base_path, f"client_{client_id + 1}")
            os.makedirs(client_dir, exist_ok=True)
            
            for split_name, dataset in data_splits.items():
                file_path = os.path.join(client_dir, f"{split_name}.json")
                
                # Convert to JSON format
                json_data = []
                for example in dataset:
                    json_data.append(example)
                
                with open(file_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                
                logger.info(f"Saved {len(json_data)} examples to {file_path}")

def prepare_medical_qa_data(
    dataset_name: str = "medqa",
    num_clients: int = 3,
    split_strategy: str = "specialty_based"
):
    """Prepare real medical QA data for federated learning."""
    
    manager = MedicalQADatasetManager()
    
    # Load dataset
    train_dataset = manager.load_dataset(dataset_name, "train")
    
    logger.info(f"Loaded {len(train_dataset)} training examples from {dataset_name}")
    
    # Create federated splits
    federated_splits = manager.create_federated_splits(
        train_dataset, num_clients, split_strategy
    )
    
    # Save to disk
    manager.save_federated_data(federated_splits)
    
    # Print statistics
    print(f"\nFederated data preparation completed for {dataset_name}")
    print(f"Split strategy: {split_strategy}")
    print(f"Number of clients: {num_clients}")
    
    for client_id, splits in federated_splits.items():
        train_size = len(splits["train"])
        val_size = len(splits["val"])
        print(f"Client {client_id + 1}: {train_size} train, {val_size} val samples")

if __name__ == "__main__":
    prepare_medical_qa_data()
