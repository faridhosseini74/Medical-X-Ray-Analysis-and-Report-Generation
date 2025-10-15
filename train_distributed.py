#!/usr/bin/env python3
"""
Distributed training script optimized for multi-GPU setup.
Supports DataParallel and DistributedDataParallel training.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime
from pathlib import Path

from models import ViTClassifier, SwinV2Classifier, DINOv2Classifier, DINOv2RegistersClassifier
from models.dinov2_model import DINOv3Classifier
from utils.dataset import create_dataloaders, CheXpertDataset, get_transforms
from utils.metrics import MetricsTracker
from utils.visualization import plot_training_history


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Distributed training for X-ray classification')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, default='vit',
                        choices=['vit', 'swin_v2', 'dinov2', 'dinov2_registers', 'dinov3', 'all'],
                        help='Model architecture to train')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    
    # Multi-GPU arguments
    parser.add_argument('--world-size', type=int, default=3,
                        help='Number of GPUs to use')
    parser.add_argument('--dist-url', type=str, default='env://',
                        help='URL for distributed training')
    parser.add_argument('--dist-backend', type=str, default='nccl',
                        help='Distributed backend')
    parser.add_argument('--rank', type=int, default=0,
                        help='Node rank for distributed training')
    
    return parser.parse_args()


def setup_distributed(rank, world_size, backend='nccl'):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


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


def train_distributed(rank, world_size, args, config, model_name):
    """Main distributed training function."""
    
    # Setup distributed training
    setup_distributed(rank, world_size, config['training'].get('backend', 'nccl'))
    
    device = torch.device(f'cuda:{rank}')
    
    # Get model-specific image size
    model_config = config['models'][model_name]
    model_image_size = model_config.get('image_size', config['dataset']['image_size'])
    
    # Temporarily update config with model-specific image size
    original_image_size = config['dataset']['image_size']
    config['dataset']['image_size'] = model_image_size
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Distributed Training: {model_name.upper()}")
        print(f"{'='*80}")
        print(f"World Size: {world_size} GPUs")
        print(f"Global Batch Size: {config['training']['batch_size'] * world_size}")
        print(f"Workers per GPU: {config['training']['num_workers']}")
        print(f"Image Size: {model_image_size}x{model_image_size}")
        print(f"{'='*80}\n")
    
    # Create model
    model = create_model(model_name, config, device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Load data with distributed sampler
    data_dir = config['dataset']['data_dir']
    train_csv = config['dataset']['csv_path']
    val_csv = config['dataset']['val_csv_path']
    
    # Create datasets
    train_transform = get_transforms(config, mode='train')
    val_transform = get_transforms(config, mode='val')
    
    train_dataset = CheXpertDataset(
        csv_path=train_csv,
        data_dir=data_dir,
        transform=train_transform,
        policy='ones',
        frontal_only=True
    )
    
    val_dataset = CheXpertDataset(
        csv_path=val_csv,
        data_dir=data_dir,
        transform=val_transform,
        policy='zeros',
        frontal_only=True
    )
    
    # Distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['training']['num_workers'] > 0 else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['training']['num_workers'] > 0 else False
    )
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer with learning rate scaling for multi-GPU
    base_lr = config['training']['learning_rate']
    scaled_lr = base_lr * world_size  # Linear scaling rule
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs']
    )
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if config['training']['mixed_precision'] else None
    
    # TensorBoard (only on rank 0)
    if rank == 0:
        log_dir = os.path.join('runs', model_name, datetime.now().strftime('%Y%m%d-%H%M%S'))
        writer = SummaryWriter(log_dir)
        checkpoint_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_val_auc = 0.0
    labels = config['dataset']['labels']
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        # Set epoch for sampler
        train_sampler.set_epoch(epoch)
        
        # Train
        model.train()
        train_metrics = MetricsTracker(labels)
        
        if rank == 0:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config["training"]["num_epochs"]} [Train]')
        else:
            pbar = train_loader
        
        for batch in pbar:
            images = batch['image'].to(device, non_blocking=True)
            targets = batch['labels'].to(device, non_blocking=True)
            
            # Forward pass
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Metrics
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            train_metrics.update(targets, preds, probs, loss.item())
            
            if rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validate
        model.eval()
        val_metrics = MetricsTracker(labels)
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device, non_blocking=True)
                targets = batch['labels'].to(device, non_blocking=True)
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                val_metrics.update(targets, preds, probs, loss.item())
        
        # Compute metrics
        train_results = train_metrics.compute()
        val_results = val_metrics.compute()
        
        # Synchronize metrics across GPUs
        train_auc = torch.tensor(train_results.get('macro_auc', 0.0)).to(device)
        val_auc = torch.tensor(val_results.get('macro_auc', 0.0)).to(device)
        train_acc = torch.tensor(train_results.get('accuracy', 0.0)).to(device)
        val_acc = torch.tensor(val_results.get('accuracy', 0.0)).to(device)
        train_f1 = torch.tensor(train_results.get('f1_macro', 0.0)).to(device)
        val_f1 = torch.tensor(val_results.get('f1_macro', 0.0)).to(device)
        
        dist.all_reduce(train_auc, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_auc, op=dist.ReduceOp.AVG)
        dist.all_reduce(train_acc, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_acc, op=dist.ReduceOp.AVG)
        dist.all_reduce(train_f1, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_f1, op=dist.ReduceOp.AVG)
        
        train_results['macro_auc'] = train_auc.item()
        val_results['macro_auc'] = val_auc.item()
        train_results['accuracy'] = train_acc.item()
        val_results['accuracy'] = val_acc.item()
        train_results['f1_macro'] = train_f1.item()
        val_results['f1_macro'] = val_f1.item()
        
        # Logging (only on rank 0)
        if rank == 0:
            print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
            print(f"Train Loss: {train_results['loss']:.4f}, Train AUC: {train_results.get('macro_auc', 0.0):.4f}")
            print(f"Train Acc: {train_results.get('accuracy', 0.0):.4f}, F1: {train_results.get('f1_macro', 0.0):.4f}, "
                  f"Prec: {train_results.get('precision_macro', 0.0):.4f}, Rec: {train_results.get('recall_macro', 0.0):.4f}")
            print(f"Val Loss: {val_results['loss']:.4f}, Val AUC: {val_results.get('macro_auc', 0.0):.4f}")
            print(f"Val Acc: {val_results.get('accuracy', 0.0):.4f}, F1: {val_results.get('f1_macro', 0.0):.4f}, "
                  f"Prec: {val_results.get('precision_macro', 0.0):.4f}, Rec: {val_results.get('recall_macro', 0.0):.4f}")
            
            writer.add_scalar('Loss/train', train_results['loss'], epoch)
            writer.add_scalar('Loss/val', val_results['loss'], epoch)
            writer.add_scalar('AUC/train', train_results.get('macro_auc', 0.0), epoch)
            writer.add_scalar('AUC/val', val_results.get('macro_auc', 0.0), epoch)
            writer.add_scalar('Accuracy/train', train_results.get('accuracy', 0.0), epoch)
            writer.add_scalar('Accuracy/val', val_results.get('accuracy', 0.0), epoch)
            writer.add_scalar('F1/train', train_results.get('f1_macro', 0.0), epoch)
            writer.add_scalar('F1/val', val_results.get('f1_macro', 0.0), epoch)
            writer.add_scalar('Precision/train', train_results.get('precision_macro', 0.0), epoch)
            writer.add_scalar('Precision/val', val_results.get('precision_macro', 0.0), epoch)
            writer.add_scalar('Recall/train', train_results.get('recall_macro', 0.0), epoch)
            writer.add_scalar('Recall/val', val_results.get('recall_macro', 0.0), epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            # Use F1 if AUC is NaN or 0
            current_metric = val_results.get('macro_auc', 0.0)
            if current_metric == 0.0 or np.isnan(current_metric):
                current_metric = val_results.get('f1_macro', 0.0)
            
            is_best = current_metric > best_val_auc
            if is_best:
                best_val_auc = current_metric
                checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_auc': best_val_auc,
                }, checkpoint_path)
                print(f"✓ Saved best model (AUC: {best_val_auc:.4f})")
        
        scheduler.step()
    
    # Restore original image size
    config['dataset']['image_size'] = original_image_size
    
    if rank == 0:
        writer.close()
        print(f"\nTraining completed! Best Val AUC: {best_val_auc:.4f}")
    
    cleanup_distributed()


def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    world_size = args.world_size
    
    if world_size > 1:
        # Distributed training
        mp.spawn(
            train_distributed,
            args=(world_size, args, config, args.model),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU training
        print("Single GPU training - use train.py instead for better experience")
        train_distributed(0, 1, args, config, args.model)


if __name__ == '__main__':
    main()
