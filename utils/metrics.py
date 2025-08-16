import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import torch
import logging

logger = logging.getLogger(__name__)

class QAMetricsTracker:
    """Metrics tracker for medical question-answering tasks."""
    
    def __init__(self, qa_format: str = "multiple_choice"):
        self.qa_format = qa_format
        self.metrics_history = []
    
    def extract_predictions_and_labels(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[List[Any], List[Any]]:
        """Extract predictions and labels based on QA format."""
        
        if self.qa_format == "multiple_choice":
            return self._extract_mc_predictions(outputs, batch)
        elif self.qa_format == "extractive":
            return self._extract_extractive_predictions(outputs, batch)
        else:  # generative
            return self._extract_generative_predictions(outputs, batch)
    
    def _extract_mc_predictions(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[List[int], List[int]]:
        """Extract predictions for multiple choice QA."""
        
        logits = outputs['logits']
        labels = batch['labels']
        
        # Get predicted choices
        predictions = torch.argmax(logits, dim=-1).cpu().tolist()
        true_labels = labels.cpu().tolist()
        
        return predictions, true_labels
    
    def _extract_extractive_predictions(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Extract predictions for extractive QA."""
        
        start_logits = outputs['start_logits']
        end_logits = outputs['end_logits']
        
        start_positions = batch.get('start_positions')
        end_positions = batch.get('end_positions')
        
        # Get predicted spans
        pred_starts = torch.argmax(start_logits, dim=-1).cpu().tolist()
        pred_ends = torch.argmax(end_logits, dim=-1).cpu().tolist()
        
        predictions = list(zip(pred_starts, pred_ends))
        
        if start_positions is not None and end_positions is not None:
            true_starts = start_positions.cpu().tolist()
            true_ends = end_positions.cpu().tolist()
            labels = list(zip(true_starts, true_ends))
        else:
            labels = [(0, 0)] * len(predictions)  # Dummy labels
        
        return predictions, labels
    
    def _extract_generative_predictions(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[List[str], List[str]]:
        """Extract predictions for generative QA."""
        
        logits = outputs['logits']
        labels = batch.get('labels')
        
        # Get predicted tokens (greedy decoding)
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Convert to text (this is simplified - would need tokenizer)
        predictions = predicted_ids.cpu().tolist()
        
        if labels is not None:
            true_labels = labels.cpu().tolist()
        else:
            true_labels = [0] * len(predictions)  # Dummy labels
        
        return predictions, true_labels
    
    def compute_metrics(
        self, 
        predictions: List[Any], 
        labels: List[Any]
    ) -> Tuple[float, float]:
        """Compute accuracy and F1 score based on QA format."""
        
        if self.qa_format == "multiple_choice":
            return self._compute_mc_metrics(predictions, labels)
        elif self.qa_format == "extractive":
            return self._compute_extractive_metrics(predictions, labels)
        else:  # generative
            return self._compute_generative_metrics(predictions, labels)
    
    def _compute_mc_metrics(
        self, 
        predictions: List[int], 
        labels: List[int]
    ) -> Tuple[float, float]:
        """Compute metrics for multiple choice QA."""
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        return accuracy, f1
    
    def _compute_extractive_metrics(
        self, 
        predictions: List[Tuple[int, int]], 
        labels: List[Tuple[int, int]]
    ) -> Tuple[float, float]:
        """Compute metrics for extractive QA."""
        
        # Exact match accuracy
        exact_matches = [pred == label for pred, label in zip(predictions, labels)]
        accuracy = np.mean(exact_matches)
        
        # Overlap F1 (simplified)
        f1_scores = []
        for (pred_start, pred_end), (true_start, true_end) in zip(predictions, labels):
            overlap_start = max(pred_start, true_start)
            overlap_end = min(pred_end, true_end)
            
            if overlap_start <= overlap_end:
                overlap_len = overlap_end - overlap_start + 1
                pred_len = pred_end - pred_start + 1
                true_len = true_end - true_start + 1
                
                precision = overlap_len / max(pred_len, 1)
                recall = overlap_len / max(true_len, 1)
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
            else:
                f1 = 0.0
            
            f1_scores.append(f1)
        
        f1 = np.mean(f1_scores)
        
        return accuracy, f1
    
    def _compute_generative_metrics(
        self, 
        predictions: List[str], 
        labels: List[str]
    ) -> Tuple[float, float]:
        """Compute metrics for generative QA."""
        
        # Simplified metrics for generative QA
        # In practice, would use BLEU, ROUGE, or semantic similarity
        
        exact_matches = [pred == label for pred, label in zip(predictions, labels)]
        accuracy = np.mean(exact_matches)
        
        # Simplified F1 based on token overlap
        f1_scores = []
        for pred, label in zip(predictions, labels):
            if isinstance(pred, list) and isinstance(label, list):
                pred_set = set(pred)
                label_set = set(label)
                
                if len(pred_set) == 0 and len(label_set) == 0:
                    f1 = 1.0
                elif len(pred_set) == 0 or len(label_set) == 0:
                    f1 = 0.0
                else:
                    overlap = len(pred_set & label_set)
                    precision = overlap / len(pred_set)
                    recall = overlap / len(label_set)
                    
                    if precision + recall > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                    else:
                        f1 = 0.0
            else:
                f1 = 1.0 if pred == label else 0.0
            
            f1_scores.append(f1)
        
        f1 = np.mean(f1_scores)
        
        return accuracy, f1
    
    def log_metrics(
        self, 
        round_num: int, 
        metrics: Dict[str, float]
    ):
        """Log metrics for a training round."""
        
        metrics_entry = {
            'round': round_num,
            'timestamp': np.datetime64('now'),
            **metrics
        }
        
        self.metrics_history.append(metrics_entry)
        
        logger.info(f"Round {round_num} metrics: {metrics}")
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best metrics across all rounds."""
        
        if not self.metrics_history:
            return {}
        
        best_accuracy = max(entry.get('eval_accuracy', 0) for entry in self.metrics_history)
        best_f1 = max(entry.get('eval_f1', 0) for entry in self.metrics_history)
        
        return {
            'best_accuracy': best_accuracy,
            'best_f1': best_f1
        }

    def get_convergence_round(self, threshold: float = 0.01) -> Optional[int]:
        """Get round number where model converged (accuracy change < threshold)."""
        
        if len(self.metrics_history) < 5:  # Need at least 5 rounds to detect convergence
            return None
        
        accuracies = [entry.get('eval_accuracy', 0) for entry in self.metrics_history]
        
        for i in range(4, len(accuracies)):
            # Check if accuracy change is small for last 3 rounds
            recent_changes = [abs(accuracies[j] - accuracies[j-1]) for j in range(i-2, i+1)]
            if all(change < threshold for change in recent_changes):
                return i + 1  # Round numbers are 1-indexed
        
        return None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all metrics."""
        
        if not self.metrics_history:
            return {"status": "no_data", "message": "No metrics recorded yet"}
        
        # Extract metric values
        accuracies = [entry.get('eval_accuracy', 0) for entry in self.metrics_history]
        f1_scores = [entry.get('eval_f1', 0) for entry in self.metrics_history]
        losses = [entry.get('eval_loss', 0) for entry in self.metrics_history]
        
        # Calculate statistics
        summary = {
            'total_rounds': len(self.metrics_history),
            'qa_format': self.qa_format,
            'accuracy_stats': {
                'final': accuracies[-1] if accuracies else 0,
                'best': max(accuracies) if accuracies else 0,
                'worst': min(accuracies) if accuracies else 0,
                'mean': np.mean(accuracies) if accuracies else 0,
                'std': np.std(accuracies) if accuracies else 0,
                'improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0
            },
            'f1_stats': {
                'final': f1_scores[-1] if f1_scores else 0,
                'best': max(f1_scores) if f1_scores else 0,
                'worst': min(f1_scores) if f1_scores else 0,
                'mean': np.mean(f1_scores) if f1_scores else 0,
                'std': np.std(f1_scores) if f1_scores else 0,
                'improvement': f1_scores[-1] - f1_scores[0] if len(f1_scores) > 1 else 0
            },
            'loss_stats': {
                'final': losses[-1] if losses else 0,
                'best': min(losses) if losses else float('inf'),
                'worst': max(losses) if losses else 0,
                'mean': np.mean(losses) if losses else 0,
                'std': np.std(losses) if losses else 0,
                'improvement': losses[0] - losses[-1] if len(losses) > 1 else 0  # Lower is better
            },
            'convergence_info': {
                'converged': self.get_convergence_round() is not None,
                'convergence_round': self.get_convergence_round(),
                'convergence_threshold': 0.01
            }
        }
        
        return summary
    
    def plot_metrics_history(self, save_path: Optional[str] = None) -> None:
        """Plot training metrics history."""
        
        try:
            import matplotlib.pyplot as plt
            
            if not self.metrics_history:
                logger.warning("No metrics to plot")
                return
            
            # Extract data
            rounds = [entry['round'] for entry in self.metrics_history]
            accuracies = [entry.get('eval_accuracy', 0) for entry in self.metrics_history]
            f1_scores = [entry.get('eval_f1', 0) for entry in self.metrics_history]
            losses = [entry.get('eval_loss', 0) for entry in self.metrics_history]
            
            # Create subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Accuracy plot
            axes[0].plot(rounds, accuracies, marker='o', linewidth=2, color='blue')
            axes[0].set_title(f'{self.qa_format.title()} QA - Accuracy')
            axes[0].set_xlabel('Round')
            axes[0].set_ylabel('Accuracy')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, 1)
            
            # F1 Score plot
            axes[1].plot(rounds, f1_scores, marker='s', linewidth=2, color='green')
            axes[1].set_title(f'{self.qa_format.title()} QA - F1 Score')
            axes[1].set_xlabel('Round')
            axes[1].set_ylabel('F1 Score')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 1)
            
            # Loss plot
            axes[2].plot(rounds, losses, marker='^', linewidth=2, color='red')
            axes[2].set_title(f'{self.qa_format.title()} QA - Loss')
            axes[2].set_xlabel('Round')
            axes[2].set_ylabel('Loss')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Metrics plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping plot generation")
        except Exception as e:
            logger.error(f"Error plotting metrics: {e}")
    
    def export_metrics(self, file_path: str) -> None:
        """Export metrics history to JSON file."""
        
        import json
        
        try:
            # Convert numpy datetime to string for JSON serialization
            exportable_history = []
            for entry in self.metrics_history:
                export_entry = entry.copy()
                if 'timestamp' in export_entry:
                    export_entry['timestamp'] = str(export_entry['timestamp'])
                exportable_history.append(export_entry)
            
            export_data = {
                'qa_format': self.qa_format,
                'metrics_history': exportable_history,
                'summary': self.get_metrics_summary()
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Metrics exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
    
    def compare_with_baseline(self, baseline_metrics: Dict[str, float]) -> Dict[str, float]:
        """Compare current performance with baseline metrics."""
        
        if not self.metrics_history:
            return {"error": "No metrics available for comparison"}
        
        current_metrics = self.get_metrics_summary()
        
        comparison = {
            'accuracy_improvement': current_metrics['accuracy_stats']['final'] - baseline_metrics.get('accuracy', 0),
            'f1_improvement': current_metrics['f1_stats']['final'] - baseline_metrics.get('f1', 0),
            'loss_improvement': baseline_metrics.get('loss', float('inf')) - current_metrics['loss_stats']['final'],
            'performance_ratio': {
                'accuracy': current_metrics['accuracy_stats']['final'] / max(baseline_metrics.get('accuracy', 0.001), 0.001),
                'f1': current_metrics['f1_stats']['final'] / max(baseline_metrics.get('f1', 0.001), 0.001)
            }
        }
        
        # Determine overall improvement
        accuracy_better = comparison['accuracy_improvement'] > 0
        f1_better = comparison['f1_improvement'] > 0
        loss_better = comparison['loss_improvement'] > 0
        
        if all([accuracy_better, f1_better, loss_better]):
            comparison['overall_result'] = "significant_improvement"
        elif sum([accuracy_better, f1_better, loss_better]) >= 2:
            comparison['overall_result'] = "improvement"
        elif sum([accuracy_better, f1_better, loss_better]) == 1:
            comparison['overall_result'] = "mixed"
        else:
            comparison['overall_result'] = "degradation"
        
        return comparison
    
    def reset_metrics(self) -> None:
        """Reset all stored metrics."""
        self.metrics_history = []
        logger.info("Metrics history reset")
    
    def get_recent_trend(self, num_rounds: int = 5) -> Dict[str, str]:
        """Analyze trend in recent performance."""
        
        if len(self.metrics_history) < num_rounds:
            return {"trend": "insufficient_data", "message": f"Need at least {num_rounds} rounds"}
        
        recent_entries = self.metrics_history[-num_rounds:]
        recent_accuracies = [entry.get('eval_accuracy', 0) for entry in recent_entries]
        recent_f1s = [entry.get('eval_f1', 0) for entry in recent_entries]
        
        # Calculate trends
        accuracy_trend = "improving" if recent_accuracies[-1] > recent_accuracies[0] else "declining"
        f1_trend = "improving" if recent_f1s[-1] > recent_f1s[0] else "declining"
        
        # Check for plateau
        accuracy_variance = np.var(recent_accuracies)
        f1_variance = np.var(recent_f1s)
        
        if accuracy_variance < 0.001 and f1_variance < 0.001:
            overall_trend = "plateau"
        elif accuracy_trend == "improving" and f1_trend == "improving":
            overall_trend = "improving"
        elif accuracy_trend == "declining" and f1_trend == "declining":
            overall_trend = "declining"
        else:
            overall_trend = "mixed"
        
        return {
            "overall_trend": overall_trend,
            "accuracy_trend": accuracy_trend,
            "f1_trend": f1_trend,
            "accuracy_variance": float(accuracy_variance),
            "f1_variance": float(f1_variance),
            "num_rounds_analyzed": num_rounds
        }
