# Medical X-Ray Analysis System
# Data loading and preprocessing utilities

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict
from torchvision import transforms


class CheXpertDataset(Dataset):
    """
    CheXpert Dataset loader for chest X-ray images.
    Handles multi-label classification with uncertain labels.
    """
    
    # CheXpert pathology labels
    LABELS = [
        'No Finding',
        'Enlarged Cardiomediastinum',
        'Cardiomegaly',
        'Lung Opacity',
        'Lung Lesion',
        'Edema',
        'Consolidation',
        'Pneumonia',
        'Atelectasis',
        'Pneumothorax',
        'Pleural Effusion',
        'Pleural Other',
        'Fracture',
        'Support Devices'
    ]
    
    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        policy: str = 'ones',  # How to handle uncertain labels: 'ones', 'zeros', or 'ignore'
        frontal_only: bool = True
    ):
        """
        Args:
            csv_path: Path to the CSV file with image paths and labels
            data_dir: Root directory containing the images
            transform: torchvision transform pipeline
            policy: How to handle uncertain labels (-1 values)
            frontal_only: Whether to use only frontal view images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.policy = policy
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Filter for frontal views only (PA or AP)
        if frontal_only and 'Frontal/Lateral' in self.df.columns:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']
        
        # Handle missing values and uncertain labels
        self._process_labels()
        
        print(f"Loaded {len(self.df)} images from {csv_path}")
        print(f"Label distribution:")
        for label in self.LABELS:
            if label in self.df.columns:
                pos_count = (self.df[label] == 1.0).sum()
                print(f"  {label}: {pos_count} positive cases")
    
    def _process_labels(self):
        """Process labels according to the specified policy."""
        for label in self.LABELS:
            if label not in self.df.columns:
                self.df[label] = 0.0
                continue
            
            # Fill NaN with 0
            self.df[label] = self.df[label].fillna(0.0)
            
            # Handle uncertain labels (-1)
            # IMPORTANT: For CheXpert, uncertain labels are common
            # Best practice: treat uncertain as negative (zeros) to reduce noise
            if self.policy == 'ones':
                self.df.loc[self.df[label] == -1.0, label] = 1.0
            elif self.policy == 'zeros':
                self.df.loc[self.df[label] == -1.0, label] = 0.0
            elif self.policy == 'ignore':
                # Mark uncertain as NaN for potential masking during training
                self.df.loc[self.df[label] == -1.0, label] = np.nan
            else:  # default to zeros
                self.df.loc[self.df[label] == -1.0, label] = 0.0
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
                - image: Tensor of shape (C, H, W)
                - labels: Tensor of shape (num_classes,)
                - path: Image file path
        """
        row = self.df.iloc[idx]
        
        # Get image path
        img_path = row['Path'] if 'Path' in row else row.iloc[0]
        full_path = os.path.join(self.data_dir, img_path)
        
        # Load image
        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            # Return a blank image in case of error
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        else:
            # Default: just convert to tensor and normalize
            image = transforms.ToTensor()(image)
        
        # Get labels
        labels = []
        for label in self.LABELS:
            if label in row:
                labels.append(row[label])
            else:
                labels.append(0.0)
        
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return {
            'image': image,
            'labels': labels,
            'path': img_path
        }


def get_transforms(config: dict, mode: str = 'train') -> transforms.Compose:
    """
    Get augmentation transforms based on configuration.
    
    Args:
        config: Configuration dictionary
        mode: 'train' or 'val'
    
    Returns:
        torchvision composition of transforms
    """
    image_size = config['dataset']['image_size']
    
    if mode == 'train':
        aug_config = config['augmentation']['train']
        transform_list = [
            transforms.Resize((image_size, image_size)),
        ]
        
        # Add augmentations
        if aug_config.get('horizontal_flip', 0) > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=aug_config['horizontal_flip']))
        
        if aug_config.get('rotation_limit', 0) > 0:
            transform_list.append(transforms.RandomRotation(
                degrees=aug_config['rotation_limit']
            ))
        
        if aug_config.get('brightness_contrast', 0) > 0:
            transform_list.append(transforms.ColorJitter(
                brightness=aug_config['brightness_contrast'],
                contrast=aug_config['brightness_contrast']
            ))
        
        # Normalization
        norm = aug_config['normalize']
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=norm['mean'], std=norm['std'])
        ])
        
    else:  # validation/test
        aug_config = config['augmentation']['val']
        norm = aug_config['normalize']
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm['mean'], std=norm['std'])
        ]
    
    return transforms.Compose(transform_list)


def create_dataloaders(
    config: dict,
    train_csv: Optional[str] = None,
    val_csv: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        config: Configuration dictionary
        train_csv: Path to training CSV (optional, uses config if not provided)
        val_csv: Path to validation CSV (optional, uses config if not provided)
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_dir = config['dataset']['data_dir']
    train_csv = train_csv or config['dataset']['csv_path']
    val_csv = val_csv or config['dataset']['val_csv_path']
    
    # Get transforms
    train_transform = get_transforms(config, mode='train')
    val_transform = get_transforms(config, mode='val')
    
    # Create datasets
    train_dataset = CheXpertDataset(
        csv_path=train_csv,
        data_dir=data_dir,
        transform=train_transform,
        policy='zeros',  # IMPORTANT: Treat uncertain labels as negative to reduce noise
        frontal_only=True
    )
    
    val_dataset = CheXpertDataset(
        csv_path=val_csv,
        data_dir=data_dir,
        transform=val_transform,
        policy='zeros',  # Convert uncertain to negative for validation
        frontal_only=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset loader
    import yaml
    
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create a sample dataset
    transform = get_transforms(config, mode='train')
    
    print("Dataset loader test completed successfully!")
