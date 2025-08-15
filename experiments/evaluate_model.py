import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse
import logging
from typing import Dict, Any, List
from sklearn.metrics import classification_report, confusion_matrix

from ..config.client_config import ClientConfig
from ..client.federated_client import FederatedQAClient
from ..utils.metrics import QAMetricsTracker

logger = logging.getLogger(__name__)

class MedicalQAEvaluator:
    """Comprehensive evaluator for federated medical QA models."""
    
    def __init__(self, client_config: ClientConfig):
        self.config = client_config
        self.device = torch.device(client_config.device)
        
        # Initialize client for model access
        self.client = FederatedQAClient(client_config)
        self.metrics_tracker = QAMetricsTracker(client_config.qa_format)
        
    def evaluate_on_test_set(self) -> Dict[str, Any]:
        """Comprehensive evaluation on test/validation set."""
        
        self.client.client_model.eval()
        
        all_predictions = []
        all_labels = []
        all_losses = []
        detailed_results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.client.val_loader):
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.client.client_model.forward_local(**batch)
                
                # Extract predictions and labels
                predictions, labels = self.metrics_tracker.extract_predictions_and_labels(
                    outputs, batch
                )
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                
                if 'loss' in outputs:
                    all_losses.append(outputs['loss'].item())
                
                # Store detailed results for analysis
                for i, (pred, label) in enumerate(zip(predictions, labels)):
                    detailed_results.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'prediction': pred,
                        'label': label,
                        'correct': pred == label,
                        'qa_format': self.config.qa_format
                    })
        
        # Calculate overall metrics
        accuracy, f1 = self.metrics_tracker.compute_metrics(all_predictions, all_labels)
        avg_loss = np.mean(all_losses) if all_losses else 0.0
        
        # Calculate additional metrics for medical QA
        medical_metrics = self._calculate_medical_metrics(all_predictions, all_labels)
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'avg_loss': avg_loss,
            'num_samples': len(all_predictions),
            'detailed_results': detailed_results,
            **medical_metrics
        }
        
        return results
    
    def _calculate_medical_metrics(
        self, 
        predictions: List[Any], 
        labels: List[Any]
    ) -> Dict[str, float]:
        """Calculate medical domain-specific metrics."""
        
        metrics = {}
        
        if self.config.qa_format == "multiple_choice":
            # For multiple choice, calculate per-class metrics
            if len(set(labels)) > 1:  # More than one class
                from sklearn.metrics import precision_recall_fscore_support
                
                precision, recall, f1, support = precision_recall_fscore_support(
                    labels, predictions, average='weighted', zero_division=0
                )
                
                metrics.update({
                    'weighted_precision': precision,
                    'weighted_recall': recall,
                    'weighted_f1': f1
                })
                
                # Calculate macro averages
                precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                    labels, predictions, average='macro', zero_division=0
                )
                
                metrics.update({
                    'macro_precision': precision_macro,
                    'macro_recall': recall_macro,
                    'macro_f1': f1_macro
                })
        
        elif self.config.qa_format == "extractive":
            # For extractive QA, calculate span overlap metrics
            exact_matches = sum(1 for p, l in zip(predictions, labels) if p == l)
            metrics['exact_match'] = exact_matches / len(predictions)
            
            # Calculate token-level F1 (simplified)
            token_f1_scores = []
            for (pred_start, pred_end), (true_start, true_end) in zip(predictions, labels):
                overlap_start = max(pred_start, true_start)
                overlap_end = min(pred_end, true_end)
                
                if overlap_start <= overlap_end:
                    overlap_tokens = overlap_end - overlap_start + 1
                    pred_tokens = pred_end - pred_start + 1
                    true_tokens = true_end - true_start + 1
                    
                    if pred_tokens > 0 and true_tokens > 0:
                        precision = overlap_tokens / pred_tokens
                        recall = overlap_tokens / true_tokens
                        
                        if precision + recall > 0:
                            f1 = 2 * precision * recall / (precision + recall)
                        else:
                            f1 = 0.0
                    else:
                        f1 = 0.0
                else:
                    f1 = 0.0
                
                token_f1_scores.append(f1)
            
            metrics['token_f1'] = np.mean(token_f1_scores)
        
        # Calculate confidence metrics if applicable
        if hasattr(self.client.client_model, 'get_prediction_confidence'):
            confidences = self.client.client_model.get_prediction_confidence()
            metrics['avg_confidence'] = np.mean(confidences)
            metrics['confidence_std'] = np.std(confidences)
        
        return metrics
    
    def analyze_error_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error patterns in medical QA predictions."""
        
        detailed_results = results['detailed_results']
        
        # Error analysis
        correct_predictions = [r for r in detailed_results if r['correct']]
        incorrect_predictions = [r for r in detailed_results if not r['correct']]
        
        error_analysis = {
            'total_samples': len(detailed_results),
            'correct_count': len(correct_predictions),
            'incorrect_count': len(incorrect_predictions),
            'error_rate': len(incorrect_predictions) / len(detailed_results)
        }
        
        if self.config.qa_format == "multiple_choice":
            # Analyze confusion patterns
            all_predictions = [r['prediction'] for r in detailed_results]
            all_labels = [r['label'] for r in detailed_results]
            
            if len(set(all_labels)) > 1:
                cm = confusion_matrix(all_labels, all_predictions)
                error_analysis['confusion_matrix'] = cm.tolist()
                
                # Most confused classes
                np.fill_diagonal(cm, 0)  # Remove correct predictions
                max_confusion_idx = np.unravel_index(np.argmax(cm), cm.shape)
                error_analysis['most_confused_classes'] = {
                    'true_class': int(max_confusion_idx[0]),
                    'predicted_class': int(max_confusion_idx[1]),
                    'confusion_count': int(cm[max_confusion_idx])
                }
        
        return error_analysis
    
    def generate_medical_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate medical domain-specific insights."""
        
        insights = {
            'performance_summary': {
                'accuracy': results['accuracy'],
                'f1_score': results['f1_score'],
                'performance_level': self._classify_performance_level(results['accuracy'])
            }
        }
        
        # QA format specific insights
        if self.config.qa_format == "multiple_choice":
            insights['mc_insights'] = {
                'suitable_for': 'Medical board exams, diagnosis selection, treatment choice',
                'clinical_relevance': 'High - mirrors clinical decision making',
                'recommendation': self._get_mc_recommendation(results['accuracy'])
            }
        
        elif self.config.qa_format == "extractive":
            insights['extractive_insights'] = {
                'suitable_for': 'Information extraction from clinical notes, finding specific details',
                'clinical_relevance': 'Medium - useful for clinical information retrieval',
                'recommendation': self._get_extractive_recommendation(results.get('exact_match', 0))
            }
        
        else:  # generative
            insights['generative_insights'] = {
                'suitable_for': 'Clinical note generation, patient education, treatment explanations',
                'clinical_relevance': 'High - can generate human-readable medical text',
                'recommendation': self._get_generative_recommendation(results['f1_score'])
            }
        
        # Privacy impact assessment
        insights['privacy_impact'] = {
            'noise_level': self.config.gaussian_noise_std,
            'quantization_bits': self.config.quantization_bits,
            'privacy_utility_tradeoff': self._assess_privacy_utility_tradeoff(results['accuracy'])
        }
        
        return insights
    
    def _classify_performance_level(self, accuracy: float) -> str:
        """Classify performance level for medical QA."""
        if accuracy >= 0.85:
            return "Excellent - Clinical deployment ready"
        elif accuracy >= 0.75:
            return "Good - Suitable for clinical assistance"
        elif accuracy >= 0.65:
            return "Fair - Needs improvement for clinical use"
        else:
            return "Poor - Not suitable for clinical deployment"
    
    def _get_mc_recommendation(self, accuracy: float) -> str:
        """Get recommendation for multiple choice performance."""
        if accuracy >= 0.80:
            return "Model performs well on medical multiple choice. Consider integration with medical education platforms."
        elif accuracy >= 0.70:
            return "Good performance. Consider additional training on specific medical specialties."
        else:
            return "Performance needs improvement. Increase training data or adjust model architecture."
    
    def _get_extractive_recommendation(self, exact_match: float) -> str:
        """Get recommendation for extractive QA performance."""
        if exact_match >= 0.70:
            return "Excellent span extraction. Suitable for clinical information extraction tasks."
        elif exact_match >= 0.50:
            return "Good extraction capability. Consider fine-tuning on domain-specific medical texts."
        else:
            return "Extraction needs improvement. Consider data augmentation or better span labeling."
    
    def _get_generative_recommendation(self, f1_score: float) -> str:
        """Get recommendation for generative QA performance."""
        if f1_score >= 0.75:
            return "Strong generative capability. Suitable for patient education and clinical documentation."
        elif f1_score >= 0.60:
            return "Reasonable generation quality. Consider additional medical text training."
        else:
            return "Generation quality needs improvement. Consider larger model or more diverse training data."
    
    def _assess_privacy_utility_tradeoff(self, accuracy: float) -> str:
        """Assess the privacy-utility tradeoff."""
        noise_level = self.config.gaussian_noise_std
        
        if noise_level <= 0.01 and accuracy >= 0.80:
            return "Excellent - High utility with minimal privacy cost"
        elif noise_level <= 0.05 and accuracy >= 0.75:
            return "Good - Reasonable utility with low privacy cost"
        elif noise_level <= 0.1 and accuracy >= 0.65:
            return "Fair - Moderate utility with moderate privacy protection"
        else:
            return "Poor - Privacy protection may be too costly for utility"
    
    def plot_evaluation_results(self, results: Dict[str, Any], save_path: str = None):
        """Plot comprehensive evaluation results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Performance metrics
        metrics = ['accuracy', 'f1_score', 'avg_loss']
        values = [results.get(m, 0) for m in metrics]
        
        axes[0, 0].bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 0].set_title('Overall Performance Metrics')
        axes[0, 0].set_ylabel('Score/Loss')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Confusion matrix (for multiple choice)
        if self.config.qa_format == "multiple_choice" and 'confusion_matrix' in results:
            cm = np.array(results['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues')
            axes[0, 1].set_title('Confusion Matrix')
            axes[0, 1].set_xlabel('Predicted')
            axes[0, 1].set_ylabel('Actual')
        else:
            axes[0, 1].text(0.5, 0.5, f'{self.config.qa_format.title()} QA\nResults', 
                           ha='center', va='center', transform=axes[0, 1].transAxes,
                           fontsize=14, bbox=dict(boxstyle="round", facecolor='lightgray'))
            axes[0, 1].set_title('QA Format Specific Results')
        
        # Error analysis
        detailed_results = results['detailed_results']
        correct_count = sum(1 for r in detailed_results if r['correct'])
        incorrect_count = len(detailed_results) - correct_count
        
        axes[1, 0].pie([correct_count, incorrect_count], 
                      labels=['Correct', 'Incorrect'],
                      colors=['lightgreen', 'lightcoral'],
                      autopct='%1.1f%%',
                      startangle=90)
        axes[1, 0].set_title('Prediction Accuracy Distribution')
        
        # Performance by sample (trend)
        sample_performance = [r['correct'] for r in detailed_results[:100]]  # First 100 samples
        cumulative_accuracy = np.cumsum(sample_performance) / np.arange(1, len(sample_performance) + 1)
        
        axes[1, 1].plot(cumulative_accuracy, color='blue', linewidth=2)
        axes[1, 1].set_title('Cumulative Accuracy Trend (First 100 Samples)')
        axes[1, 1].set_xlabel('Sample Number')
        axes[1, 1].set_ylabel('Cumulative Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plots saved to {save_path}")
        
        plt.close()

def generate_experiment_summary(results_dir: str, num_clients: int):
    """Generate comprehensive experiment summary."""
    
    logger.info("Generating experiment summary...")
    
    # Collect all client results
    all_results = {}
    for client_id in range(1, num_clients + 1):
        result_file = os.path.join(results_dir, f"client_{client_id}_results.json")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                all_results[f"client_{client_id}"] = json.load(f)
    
    if not all_results:
        logger.warning("No client results found for summary generation")
        return
    
    # Generate summary statistics
    summary = {
        'experiment_info': {
            'num_clients': num_clients,
            'clients_completed': len(all_results),
            'completion_rate': len(all_results) / num_clients
        },
        'performance_summary': {},
        'convergence_analysis': {},
        'federated_insights': {}
    }
    
    # Performance analysis
    final_accuracies = []
    final_f1_scores = []
    best_accuracies = []
    
    for client_id, results in all_results.items():
        if 'eval_accuracy' in results and results['eval_accuracy']:
            final_accuracies.append(results['eval_accuracy'][-1])
            best_accuracies.append(max(results['eval_accuracy']))
        
        if 'eval_f1' in results and results['eval_f1']:
            final_f1_scores.append(results['eval_f1'][-1])
    
    if final_accuracies:
        summary['performance_summary'] = {
            'avg_final_accuracy': np.mean(final_accuracies),
            'std_final_accuracy': np.std(final_accuracies),
            'min_final_accuracy': np.min(final_accuracies),
            'max_final_accuracy': np.max(final_accuracies),
            'avg_best_accuracy': np.mean(best_accuracies) if best_accuracies else 0,
            'avg_final_f1': np.mean(final_f1_scores) if final_f1_scores else 0
        }
    
    # Convergence analysis
    rounds_to_convergence = []
    for client_id, results in all_results.items():
        if 'eval_accuracy' in results and len(results['eval_accuracy']) >= 5:
            # Simple convergence detection
            accuracies = results['eval_accuracy']
            for i in range(4, len(accuracies)):
                recent_changes = [abs(accuracies[j] - accuracies[j-1]) for j in range(i-2, i+1)]
                if all(change < 0.01 for change in recent_changes):
                    rounds_to_convergence.append(i + 1)
                    break
    
    if rounds_to_convergence:
        summary['convergence_analysis'] = {
            'avg_rounds_to_convergence': np.mean(rounds_to_convergence),
            'min_rounds_to_convergence': np.min(rounds_to_convergence),
            'max_rounds_to_convergence': np.max(rounds_to_convergence),
            'convergence_rate': len(rounds_to_convergence) / len(all_results)
        }
    
    # Federated learning insights
    if len(all_results) > 1:
        # Calculate client similarity
        client_performances = [results['eval_accuracy'][-1] for results in all_results.values() 
                             if 'eval_accuracy' in results and results['eval_accuracy']]
        
        if len(client_performances) > 1:
            performance_variance = np.var(client_performances)
            summary['federated_insights'] = {
                'client_performance_variance': performance_variance,
                'performance_heterogeneity': 'High' if performance_variance > 0.01 else 'Low',
                'federation_effectiveness': 'Good' if performance_variance < 0.005 else 'Needs Improvement'
            }
    
    # Save summary
    summary_path = os.path.join(results_dir, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Experiment summary saved to {summary_path}")
    
    # Print key results
    print("\n" + "="*60)
    print("FEDERATED MEDICAL QA EXPERIMENT SUMMARY")
    print("="*60)
    
    if 'performance_summary' in summary and summary['performance_summary']:
        perf = summary['performance_summary']
        print(f"Average Final Accuracy: {perf['avg_final_accuracy']:.3f} ± {perf['std_final_accuracy']:.3f}")
        print(f"Best Client Accuracy: {perf['max_final_accuracy']:.3f}")
        print(f"Worst Client Accuracy: {        elif self.qa_format == "extractive":
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
                extended_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                extended_mask = extended_mask.expand(-1, -1, attention_mask.size(-1), -1)
                extended_mask = (1.0 - extended_mask) * -10000.0
            else:
                extended_mask = None
            
            hidden_states = layer(hidden_states, src_key_padding_mask=extended_mask)
        
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
