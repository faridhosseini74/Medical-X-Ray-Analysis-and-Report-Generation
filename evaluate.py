#!/usr/bin/env python3
# Evaluation script for trained models

import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from models import ViTClassifier, SwinV2Classifier, DINOv2Classifier, DINOv2RegistersClassifier
from models.dinov2_model import DINOv3Classifier
from utils.dataset import create_dataloaders
from utils.metrics import (
    calculate_metrics, plot_roc_curves, plot_confusion_matrix,
    print_classification_report, MetricsTracker
)
from utils.visualization import visualize_batch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained X-ray classification models')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                        choices=['vit', 'swin_v2', 'dinov2', 'dinov2_registers', 'dinov3'],
                        help='Model to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                        help='Output directory for evaluation results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run evaluation on')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    return parser.parse_args()


def load_model(model_name, checkpoint_path, config, device='cuda'):
    """Load a trained model from checkpoint."""
    num_classes = config['dataset']['num_classes']
    
    # Create model
    if model_name == 'vit':
        model_config = config['models']['vit']
        model = ViTClassifier(
            model_name=model_config['name'],
            num_classes=num_classes,
            pretrained=False
        )
    elif model_name == 'swin_v2':
        model_config = config['models']['swin_v2']
        model = SwinV2Classifier(
            model_name=model_config['name'],
            num_classes=num_classes,
            pretrained=False
        )
    elif model_name == 'dinov2':
        model_config = config['models']['dinov2']
        model = DINOv2Classifier(
            model_name=model_config['name'],
            num_classes=num_classes,
            pretrained=False
        )
    elif model_name == 'dinov2_registers':
        model_config = config['models']['dinov2_registers']
        model = DINOv2RegistersClassifier(
            model_name=model_config['name'],
            num_classes=num_classes,
            pretrained=False
        )
    elif model_name == 'dinov3':
        model_config = config['models'].get('dinov3', config['models']['dinov2'])
        model = DINOv3Classifier(
            model_name=model_config['name'],
            num_classes=num_classes,
            pretrained=False
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint info:")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val AUC: {checkpoint.get('best_val_auc', checkpoint.get('val_auc', 'N/A'))}")
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Loaded model from: {checkpoint_path}")
    
    return model.to(device)


def evaluate_model(model, dataloader, device, labels):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run on
        labels: List of label names
    
    Returns:
        Dictionary containing predictions and ground truth
    """
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_scores = []
    
    print("\nRunning evaluation...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            labels_batch = batch['labels']
            
            # Forward pass
            outputs = model(images)
            scores = torch.sigmoid(outputs)
            predictions = (scores > 0.5).float()
            
            # Collect results
            all_labels.append(labels_batch.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_scores.append(scores.cpu().numpy())
    
    # Concatenate results
    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_predictions)
    y_scores = np.vstack(all_scores)
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_scores': y_scores
    }


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading dataset...")
    train_loader, val_loader = create_dataloaders(config)
    dataloader = val_loader if args.split == 'val' else train_loader
    
    print(f"Evaluating on {args.split} split ({len(dataloader.dataset)} samples)")
    
    # Load model
    print(f"\nLoading {args.model} model...")
    model = load_model(args.model, args.checkpoint, config, device)
    
    # Evaluate
    results = evaluate_model(model, dataloader, device, config['dataset']['labels'])
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(
        results['y_true'],
        results['y_pred'],
        results['y_scores'],
        config['dataset']['labels'],
        threshold=args.threshold
    )
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nOverall Metrics:")
    print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
    print(f"  Micro AUC: {metrics['micro_auc']:.4f}")
    print(f"  Macro AP: {metrics['macro_ap']:.4f}")
    print(f"  Micro AP: {metrics['micro_ap']:.4f}")
    print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f}")
    print(f"  Mean Class Accuracy: {metrics['mean_class_accuracy']:.4f}")
    
    print(f"\nPer-Class AUC:")
    for label in config['dataset']['labels']:
        key = f'{label}_auc'
        if key in metrics:
            print(f"  {label}: {metrics[key]:.4f}")
    
    # Print classification report
    print_classification_report(
        results['y_true'],
        results['y_pred'],
        config['dataset']['labels']
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save metrics
    import json
    metrics_path = os.path.join(args.output_dir, f'{args.model}_metrics.json')
    with open(metrics_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        metrics_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in metrics.items()}
        json.dump(metrics_json, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        
        # ROC curves
        print("  - ROC curves...")
        roc_path = os.path.join(args.output_dir, f'{args.model}_roc_curves.png')
        fig_roc = plot_roc_curves(
            results['y_true'],
            results['y_scores'],
            config['dataset']['labels'],
            save_path=roc_path
        )
        
        # Confusion matrices
        print("  - Confusion matrices...")
        cm_path = os.path.join(args.output_dir, f'{args.model}_confusion_matrices.png')
        fig_cm = plot_confusion_matrix(
            results['y_true'],
            results['y_pred'],
            config['dataset']['labels'],
            save_path=cm_path
        )
        
        print(f"✓ Visualizations saved to: {args.output_dir}")
    
    print(f"\n{'='*80}")
    print("Evaluation completed successfully!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
