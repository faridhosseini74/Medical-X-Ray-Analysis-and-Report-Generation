# Evaluation metrics for multi-label classification

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from typing import Dict, List, Tuple
import torch


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    labels: List[str],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for multi-label classification.
    
    Args:
        y_true: Ground truth labels, shape (n_samples, n_classes)
        y_pred: Predicted binary labels, shape (n_samples, n_classes)
        y_scores: Predicted probabilities, shape (n_samples, n_classes)
        labels: List of class names
        threshold: Threshold for binary classification
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Overall metrics - AUC
    try:
        # Check if we have valid data for AUC calculation
        valid_classes = []
        for i in range(y_true.shape[1]):
            # Need at least one positive and one negative sample
            if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true[:, i]):
                valid_classes.append(i)
        
        if len(valid_classes) > 0:
            # Calculate only for valid classes
            y_true_valid = y_true[:, valid_classes]
            y_scores_valid = y_scores[:, valid_classes]
            metrics['macro_auc'] = roc_auc_score(y_true_valid, y_scores_valid, average='macro')
            metrics['weighted_auc'] = roc_auc_score(y_true_valid, y_scores_valid, average='weighted')
        else:
            metrics['macro_auc'] = 0.0
            metrics['weighted_auc'] = 0.0
            
        # Micro AUC (overall across all classes)
        metrics['micro_auc'] = roc_auc_score(y_true.ravel(), y_scores.ravel())
        
    except Exception as e:
        print(f"Warning: Could not calculate AUC metrics: {e}")
        metrics['macro_auc'] = 0.0
        metrics['micro_auc'] = 0.0
        metrics['weighted_auc'] = 0.0
    
    # Classification metrics (using predictions)
    try:
        # Sample-wise metrics (considering all labels per sample)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Macro-averaged metrics (average across classes)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Micro-averaged metrics (aggregate across all samples and classes)
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # Weighted-averaged metrics (weighted by support)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Sample-based metrics (for multi-label)
        metrics['samples_accuracy'] = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
        
    except Exception as e:
        print(f"Warning: Could not calculate classification metrics: {e}")
        metrics['accuracy'] = 0.0
        metrics['precision_macro'] = 0.0
        metrics['recall_macro'] = 0.0
        metrics['f1_macro'] = 0.0
    
    # Average Precision metrics
    try:
        if len(valid_classes) > 0:
            y_true_valid = y_true[:, valid_classes]
            y_scores_valid = y_scores[:, valid_classes]
            metrics['macro_ap'] = average_precision_score(y_true_valid, y_scores_valid, average='macro')
            metrics['weighted_ap'] = average_precision_score(y_true_valid, y_scores_valid, average='weighted')
        else:
            metrics['macro_ap'] = 0.0
            metrics['weighted_ap'] = 0.0
    except Exception as e:
        print(f"Warning: Could not calculate AP metrics: {e}")
        metrics['macro_ap'] = 0.0
        metrics['weighted_ap'] = 0.0
    
    # Per-class metrics
    for i, label in enumerate(labels):
        try:
            if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true[:, i]):
                auc = roc_auc_score(y_true[:, i], y_scores[:, i])
                ap = average_precision_score(y_true[:, i], y_scores[:, i])
                metrics[f'{label}_auc'] = auc
                metrics[f'{label}_ap'] = ap
        except Exception as e:
            pass  # Skip classes with issues
    
    # Exact match accuracy (all labels must be correct)
    correct = (y_pred == y_true).all(axis=1).sum()
    total = y_true.shape[0]
    metrics['exact_match_accuracy'] = correct / total if total > 0 else 0.0
    
    # Hamming accuracy (label-wise accuracy)
    metrics['hamming_accuracy'] = (y_pred == y_true).mean()
    
    return metrics


def plot_roc_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    labels: List[str],
    save_path: str = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot ROC curves for all classes.
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
        labels: List of class names
        save_path: Path to save the figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    n_classes = len(labels)
    fig, axes = plt.subplots(3, 5, figsize=figsize)
    axes = axes.ravel()
    
    for i, label in enumerate(labels):
        if i >= len(axes):
            break
            
        try:
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
            auc = roc_auc_score(y_true[:, i], y_scores[:, i])
            
            # Plot
            axes[i].plot(fpr, tpr, label=f'AUC = {auc:.3f}')
            axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.3)
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(label)
            axes[i].legend(loc='lower right')
            axes[i].grid(alpha=0.3)
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error: {str(e)[:20]}',
                        ha='center', va='center')
    
    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    save_path: str = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot confusion matrices for all classes.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted binary labels
        labels: List of class names
        save_path: Path to save the figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    n_classes = len(labels)
    fig, axes = plt.subplots(3, 5, figsize=figsize)
    axes = axes.ravel()
    
    for i, label in enumerate(labels):
        if i >= len(axes):
            break
            
        try:
            # Calculate confusion matrix
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'],
                       ax=axes[i])
            axes[i].set_title(label)
            axes[i].set_ylabel('True Label')
            axes[i].set_xlabel('Predicted Label')
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error: {str(e)[:20]}',
                        ha='center', va='center')
    
    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str]
):
    """
    Print detailed classification report for each class.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted binary labels
        labels: List of class names
    """
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    
    for i, label in enumerate(labels):
        print(f"\n{label}:")
        print("-" * 40)
        try:
            print(classification_report(
                y_true[:, i], 
                y_pred[:, i],
                target_names=['Negative', 'Positive'],
                zero_division=0
            ))
        except Exception as e:
            print(f"Error: {e}")


class MetricsTracker:
    """Track metrics during training."""
    
    def __init__(self, labels: List[str]):
        self.labels = labels
        self.reset()
    
    def reset(self):
        """Reset all stored metrics."""
        self.y_true = []
        self.y_pred = []
        self.y_scores = []
        self.losses = []
    
    def update(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        scores: torch.Tensor,
        loss: float
    ):
        """
        Update metrics with new batch.
        
        Args:
            labels: Ground truth labels
            predictions: Predicted binary labels
            scores: Predicted probabilities
            loss: Batch loss
        """
        self.y_true.append(labels.cpu().detach().numpy())
        self.y_pred.append(predictions.cpu().detach().numpy())
        self.y_scores.append(scores.cpu().detach().numpy())
        self.losses.append(loss)
    
    def compute(self, threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Args:
            threshold: Classification threshold
        
        Returns:
            Dictionary of metrics
        """
        y_true = np.vstack(self.y_true)
        y_scores = np.vstack(self.y_scores)
        y_pred = (y_scores >= threshold).astype(int)
        
        metrics = calculate_metrics(
            y_true, y_pred, y_scores,
            self.labels, threshold
        )
        
        metrics['loss'] = np.mean(self.losses)
        
        return metrics
