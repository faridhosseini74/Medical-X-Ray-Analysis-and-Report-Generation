# Visualization utilities for X-ray analysis

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import torch
from PIL import Image


def visualize_predictions(
    image: np.ndarray,
    true_labels: np.ndarray,
    pred_scores: np.ndarray,
    label_names: List[str],
    threshold: float = 0.5,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize an X-ray image with predictions and ground truth.
    
    Args:
        image: Image array (H, W, C) or (C, H, W)
        true_labels: Ground truth binary labels
        pred_scores: Predicted probabilities
        label_names: List of label names
        threshold: Classification threshold
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    # Handle tensor input
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    
    # Convert from (C, H, W) to (H, W, C) if needed
    if image.shape[0] == 3 or image.shape[0] == 1:
        image = np.transpose(image, (1, 2, 0))
    
    # Normalize image for display
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display image
    if image.shape[2] == 1:
        ax1.imshow(image.squeeze(), cmap='gray')
    else:
        ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('X-Ray Image')
    
    # Create prediction comparison
    pred_labels = (pred_scores >= threshold).astype(int)
    
    # Prepare data for bar plot
    indices = np.arange(len(label_names))
    bar_width = 0.35
    
    # Sort by prediction score for better visualization
    sort_idx = np.argsort(pred_scores)[::-1]
    sorted_labels = [label_names[i] for i in sort_idx]
    sorted_true = true_labels[sort_idx]
    sorted_pred = pred_scores[sort_idx]
    
    # Plot bars
    ax2.barh(indices, sorted_pred, bar_width, 
             label='Predicted Probability', alpha=0.8)
    ax2.barh(indices + bar_width, sorted_true, bar_width,
             label='Ground Truth', alpha=0.8)
    
    # Add threshold line
    ax2.axvline(x=threshold, color='r', linestyle='--', 
                linewidth=2, label=f'Threshold ({threshold})')
    
    # Formatting
    ax2.set_yticks(indices + bar_width / 2)
    ax2.set_yticklabels(sorted_labels, fontsize=9)
    ax2.set_xlabel('Probability / Label')
    ax2.set_title('Predictions vs Ground Truth')
    ax2.legend(loc='lower right')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_xlim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history (loss and metrics over epochs).
    
    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss', 'train_auc', 'val_auc'
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    metrics = list(set([k.replace('train_', '').replace('val_', '') 
                       for k in history.keys()]))
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            ax.plot(epochs, history[train_key], 'b-', label=f'Train {metric}')
        
        if val_key in history:
            epochs = range(1, len(history[val_key]) + 1)
            ax.plot(epochs, history[val_key], 'r-', label=f'Val {metric}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} over Epochs')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_attention_map(
    image: np.ndarray,
    attention: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Create an attention heatmap overlay on the image.
    
    Args:
        image: Original image (H, W, C)
        attention: Attention weights (H', W')
        alpha: Transparency of the heatmap
    
    Returns:
        Image with attention overlay
    """
    # Resize attention to match image size using PIL
    h, w = image.shape[:2]
    attention_pil = Image.fromarray((attention * 255).astype(np.uint8))
    attention_resized = attention_pil.resize((w, h), Image.BILINEAR)
    attention = np.array(attention_resized).astype(np.float32) / 255.0
    
    # Normalize attention
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    
    # Create heatmap using matplotlib colormap
    cmap = plt.get_cmap('jet')
    heatmap = cmap(attention)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Ensure image is in correct format
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Overlay - manual weighted addition
    overlay = ((1 - alpha) * image + alpha * heatmap).astype(np.uint8)
    
    return overlay


def visualize_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    label_names: List[str],
    n_samples: int = 4,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize a batch of images with their predictions.
    
    Args:
        images: Batch of images (B, C, H, W)
        labels: Ground truth labels (B, num_classes)
        predictions: Predicted probabilities (B, num_classes)
        label_names: List of label names
        n_samples: Number of samples to visualize
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    n_samples = min(n_samples, images.shape[0])
    
    fig, axes = plt.subplots(1, n_samples, figsize=(5*n_samples, 5))
    
    if n_samples == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        # Get image
        img = images[i].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        
        # Denormalize (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # Display image
        ax.imshow(img)
        ax.axis('off')
        
        # Create title with top predictions
        pred = predictions[i].cpu().numpy()
        true = labels[i].cpu().numpy()
        
        # Get top 3 predictions
        top_indices = np.argsort(pred)[-3:][::-1]
        title_lines = []
        for idx in top_indices:
            label = label_names[idx]
            score = pred[idx]
            is_true = "✓" if true[idx] == 1 else "✗"
            title_lines.append(f"{label}: {score:.2f} {is_true}")
        
        ax.set_title('\n'.join(title_lines), fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_label_distribution(
    dataset,
    label_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the distribution of labels in the dataset.
    
    Args:
        dataset: Dataset object
        label_names: List of label names
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    # Collect all labels
    all_labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        all_labels.append(sample['labels'].numpy())
    
    all_labels = np.array(all_labels)
    
    # Count positives for each class
    positive_counts = all_labels.sum(axis=0)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(range(len(label_names)), positive_counts)
    ax.set_xticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha='right')
    ax.set_ylabel('Number of Positive Cases')
    ax.set_title('Label Distribution in Dataset')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
