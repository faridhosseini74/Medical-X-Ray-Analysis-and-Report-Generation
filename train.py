#!/usr/bin/env python3
# Training script for X-ray classification models

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from models import ViTClassifier, SwinV2Classifier, DINOv2Classifier, DINOv2RegistersClassifier
from models.dinov2_model import DINOv3Classifier
from utils.dataset import create_dataloaders
from utils.metrics import MetricsTracker, print_classification_report
from utils.visualization import plot_training_history


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train X-ray classification models')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, default='vit',
                        choices=['vit', 'swin_v2', 'dinov2', 'dinov2_registers', 'dinov3', 'all'],
                        help='Model architecture to train')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on (cuda or cpu)')
    return parser.parse_args()


class Trainer:
    """Trainer class for model training and evaluation."""
    
    def __init__(self, config, model, device='cuda'):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            model: Model to train
            device: Device to train on
        """
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        # Training config
        self.num_epochs = config['training']['num_epochs']
        self.batch_size = config['training']['batch_size']
        self.learning_rate = config['training']['learning_rate']
        
        # Calculate class weights for imbalanced dataset
        self.class_weights = self._calculate_class_weights(config)
        
        # Loss function with class weights
        if self.class_weights is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights.to(device))
            print(f"Using weighted loss with pos_weights: {self.class_weights.cpu().numpy()}")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        if config['training']['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=config['training']['weight_decay']
            )
        else:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.learning_rate
            )
        
        # Scheduler
        if config['training']['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=5
            )
        
        # Mixed precision training
        self.use_amp = config['training']['mixed_precision']
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Metrics
        self.labels = config['dataset']['labels']
        self.train_metrics = MetricsTracker(self.labels)
        self.val_metrics = MetricsTracker(self.labels)
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }
        
        # Best metric tracking
        self.best_val_auc = 0.0
        self.best_epoch = 0
        
        # Early stopping
        self.early_stopping_patience = config['training']['early_stopping']['patience']
        self.early_stopping_counter = 0
    
    def _calculate_class_weights(self, config):
        """Calculate class weights for imbalanced dataset."""
        try:
            csv_path = config['dataset']['csv_path']
            df = pd.read_csv(csv_path)
            labels = config['dataset']['labels']
            
            pos_weights = []
            for label in labels:
                if label in df.columns:
                    # Count positive and negative samples (ignore uncertain -1)
                    pos_count = (df[label] == 1.0).sum()
                    neg_count = (df[label] == 0.0).sum()
                    
                    if pos_count > 0 and neg_count > 0:
                        # Weight = neg_count / pos_count (emphasize minority class)
                        weight = neg_count / pos_count
                    else:
                        weight = 1.0
                    
                    pos_weights.append(weight)
                else:
                    pos_weights.append(1.0)
            
            return torch.tensor(pos_weights, dtype=torch.float32)
        except Exception as e:
            print(f"Could not calculate class weights: {e}")
            return None
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.num_epochs} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Calculate predictions
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Update metrics
            self.train_metrics.update(labels, preds, probs, loss.item())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute epoch metrics
        metrics = self.train_metrics.compute()
        
        return metrics
    
    def validate(self, val_loader, epoch):
        """Validate the model."""
        self.model.eval()
        self.val_metrics.reset()
        
        pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{self.num_epochs} [Val]')
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Calculate predictions
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                # Update metrics
                self.val_metrics.update(labels, preds, probs, loss.item())
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute epoch metrics
        metrics = self.val_metrics.compute()
        
        return metrics
    
    def train(self, train_loader, val_loader, checkpoint_dir, model_name):
        """Full training loop."""
        print(f"\nStarting training for {model_name}...")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Epochs: {self.num_epochs}\n")
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # TensorBoard writer
        log_dir = os.path.join('runs', model_name, datetime.now().strftime('%Y%m%d-%H%M%S'))
        writer = SummaryWriter(log_dir)
        
        for epoch in range(1, self.num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['macro_auc'])
            else:
                self.scheduler.step()
            
            # Log metrics
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train AUC: {train_metrics.get('macro_auc', 0.0):.4f}")
            print(f"Train Acc: {train_metrics.get('accuracy', 0.0):.4f}, F1: {train_metrics.get('f1_macro', 0.0):.4f}, "
                  f"Prec: {train_metrics.get('precision_macro', 0.0):.4f}, Rec: {train_metrics.get('recall_macro', 0.0):.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val AUC: {val_metrics.get('macro_auc', 0.0):.4f}")
            print(f"Val Acc: {val_metrics.get('accuracy', 0.0):.4f}, F1: {val_metrics.get('f1_macro', 0.0):.4f}, "
                  f"Prec: {val_metrics.get('precision_macro', 0.0):.4f}, Rec: {val_metrics.get('recall_macro', 0.0):.4f}")
            
            # TensorBoard logging
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('AUC/train', train_metrics.get('macro_auc', 0.0), epoch)
            writer.add_scalar('AUC/val', val_metrics.get('macro_auc', 0.0), epoch)
            writer.add_scalar('Accuracy/train', train_metrics.get('accuracy', 0.0), epoch)
            writer.add_scalar('Accuracy/val', val_metrics.get('accuracy', 0.0), epoch)
            writer.add_scalar('F1/train', train_metrics.get('f1_macro', 0.0), epoch)
            writer.add_scalar('F1/val', val_metrics.get('f1_macro', 0.0), epoch)
            writer.add_scalar('Precision/train', train_metrics.get('precision_macro', 0.0), epoch)
            writer.add_scalar('Precision/val', val_metrics.get('precision_macro', 0.0), epoch)
            writer.add_scalar('Recall/train', train_metrics.get('recall_macro', 0.0), epoch)
            writer.add_scalar('Recall/val', val_metrics.get('recall_macro', 0.0), epoch)
            writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_auc'].append(train_metrics.get('macro_auc', 0.0))
            self.history['val_auc'].append(val_metrics.get('macro_auc', 0.0))
            self.history['train_acc'].append(train_metrics.get('accuracy', 0.0))
            self.history['val_acc'].append(val_metrics.get('accuracy', 0.0))
            self.history['train_f1'].append(train_metrics.get('f1_macro', 0.0))
            self.history['val_f1'].append(val_metrics.get('f1_macro', 0.0))
            
            # Save checkpoint - use F1 if AUC is NaN or 0
            current_metric = val_metrics.get('macro_auc', 0.0)
            if current_metric == 0.0 or np.isnan(current_metric):
                current_metric = val_metrics.get('f1_macro', 0.0)
            
            is_best = current_metric > self.best_val_auc
            
            if is_best:
                self.best_val_auc = current_metric
                self.best_epoch = epoch
                self.early_stopping_counter = 0
                
                # Save best model
                checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_auc': self.best_val_auc,
                    'history': self.history
                }, checkpoint_path)
                
                print(f"✓ Saved best model (AUC: {self.best_val_auc:.4f})")
            else:
                self.early_stopping_counter += 1
            
            # Save regular checkpoint
            if epoch % self.config['training']['save_frequency'] == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_auc': val_metrics['macro_auc'],
                    'history': self.history
                }, checkpoint_path)
            
            # Early stopping
            if self.config['training']['early_stopping']['enabled']:
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    print(f"Best validation AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch}")
                    break
        
        writer.close()
        
        # Plot training history
        fig = plot_training_history(self.history)
        fig.savefig(os.path.join(checkpoint_dir, f'{model_name}_history.png'))
        
        print(f"\n{'='*80}")
        print(f"Training completed!")
        print(f"Best validation AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch}")
        print(f"{'='*80}\n")


def create_model(model_name, config, device='cuda'):
    """Create a model based on the model name."""
    num_classes = config['dataset']['num_classes']
    
    if model_name == 'vit':
        model_config = config['models']['vit']
        model = ViTClassifier(
            model_name=model_config['name'],
            num_classes=num_classes,
            pretrained=model_config['pretrained'],
            dropout=model_config['dropout']
        )
    
    elif model_name == 'swin_v2':
        model_config = config['models']['swin_v2']
        model = SwinV2Classifier(
            model_name=model_config['name'],
            num_classes=num_classes,
            pretrained=model_config['pretrained'],
            dropout=model_config['dropout']
        )
    
    elif model_name == 'dinov2':
        model_config = config['models']['dinov2']
        model = DINOv2Classifier(
            model_name=model_config['name'],
            num_classes=num_classes,
            pretrained=model_config['pretrained'],
            freeze_backbone=model_config.get('freeze_backbone', False)
        )
    
    elif model_name == 'dinov2_registers':
        model_config = config['models']['dinov2_registers']
        model = DINOv2RegistersClassifier(
            model_name=model_config['name'],
            num_classes=num_classes,
            pretrained=model_config['pretrained'],
            freeze_backbone=model_config.get('freeze_backbone', False)
        )
    
    elif model_name == 'dinov3':
        model_config = config['models'].get('dinov3', config['models']['dinov2'])
        model = DINOv3Classifier(
            model_name=model_config['name'],
            num_classes=num_classes,
            pretrained=model_config['pretrained'],
            freeze_backbone=model_config.get('freeze_backbone', False)
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Models to train
    if args.model == 'all':
        models_to_train = ['vit', 'swin_v2', 'dinov2', 'dinov2_registers', 'dinov3']
    else:
        models_to_train = [args.model]
    
    # Train each model
    for model_name in models_to_train:
        print(f"\n{'='*80}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*80}\n")
        
        # Get model-specific image size
        model_config = config['models'][model_name]
        model_image_size = model_config.get('image_size', config['dataset']['image_size'])
        
        # Temporarily update config with model-specific image size
        original_image_size = config['dataset']['image_size']
        config['dataset']['image_size'] = model_image_size
        
        print(f"Using image size: {model_image_size}x{model_image_size}")
        
        # Create data loaders for this specific model
        print("Loading dataset...")
        train_loader, val_loader = create_dataloaders(config)
        
        # Restore original image size in config
        config['dataset']['image_size'] = original_image_size
        
        # Create model
        model = create_model(model_name, config, device)
        
        # Create trainer
        trainer = Trainer(config, model, device)
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(args.output_dir, model_name)
        
        # Train
        trainer.train(train_loader, val_loader, checkpoint_dir, model_name)


if __name__ == '__main__':
    main()
