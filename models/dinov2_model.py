# DINOv2 Models for X-ray Classification

import torch
import torch.nn as nn
from typing import Optional, Literal


class DINOv2Classifier(nn.Module):
    """
    DINOv2 model for multi-label chest X-ray classification.
    Supports standard DINOv2 and DINOv2 with registers.
    """
    
    def __init__(
        self,
        model_name: str = 'dinov2_vitb14',
        num_classes: int = 14,
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        use_registers: bool = False
    ):
        """
        Args:
            model_name: Name of the DINOv2 model
            num_classes: Number of output classes
            pretrained: Whether to use pre-trained weights
            dropout: Dropout rate before the final classifier
            freeze_backbone: Whether to freeze the backbone during training
            use_registers: Whether to use DINOv2 with registers
        """
        super(DINOv2Classifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_registers = use_registers
        
        # Load DINOv2 backbone
        self.backbone = self._load_dinov2_model(model_name, pretrained, use_registers)
        
        # Get feature dimension
        self.feature_dim = self.backbone.embed_dim
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classifier head
        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim // 2, num_classes)
        )
        
        # Initialize head
        self._init_head()
    
    def _load_dinov2_model(self, model_name: str, pretrained: bool, use_registers: bool):
        """Load DINOv2 model from torch hub or timm."""
        try:
            # Try loading from timm first (if available)
            import timm
            
            if use_registers:
                model_name_timm = f'vit_base_patch14_dinov2_reg.lvd142m'
            else:
                model_name_timm = f'vit_base_patch14_dinov2.lvd142m'
            
            try:
                model = timm.create_model(
                    model_name_timm,
                    pretrained=pretrained,
                    num_classes=0
                )
                print(f"✓ Loaded {model_name_timm} from timm")
                return model
            except Exception as e:
                print(f"Failed to load from timm: {e}")
                print("Trying torch.hub...")
            
        except Exception as e:
            print(f"Could not load from timm: {e}")
            print("Attempting to load from torch.hub...")
            
        try:
            # Load from torch hub
            if use_registers:
                hub_model_name = model_name + '_reg'
            else:
                hub_model_name = model_name
            
            model = torch.hub.load('facebookresearch/dinov2', hub_model_name)
            return model
            
        except Exception as e:
            print(f"Could not load from torch.hub: {e}")
            print("Creating a custom ViT model as fallback...")
            
            # Fallback: create a basic ViT
            import timm
            model = timm.create_model(
                'vit_base_patch14_224',
                pretrained=pretrained,
                num_classes=0,
                img_size=224
            )
            return model
    
    def _init_head(self):
        """Initialize the classifier head."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Extract features
        features = self.backbone(x)
        
        # Handle different output formats
        if isinstance(features, dict):
            features = features['x_norm_clstoken']
        elif len(features.shape) == 3:  # (B, N, C)
            # Use CLS token (first token)
            features = features[:, 0]
        
        # Classify
        logits = self.head(features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            Features of shape (batch_size, feature_dim)
        """
        with torch.no_grad():
            features = self.backbone(x)
            
            if isinstance(features, dict):
                features = features['x_norm_clstoken']
            elif len(features.shape) == 3:
                features = features[:, 0]
        
        return features
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_backbone(self):
        """Freeze the backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False


class DINOv2RegistersClassifier(DINOv2Classifier):
    """DINOv2 with Registers variant."""
    
    def __init__(
        self,
        model_name: str = 'dinov2_vitb14',
        num_classes: int = 14,
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            use_registers=True
        )


class DINOv3Classifier(nn.Module):
    """
    Placeholder for DINOv3 model.
    Will use DINOv2 as fallback until DINOv3 is officially released.
    """
    
    def __init__(
        self,
        model_name: str = 'dinov3_vitb14',
        num_classes: int = 14,
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        super(DINOv3Classifier, self).__init__()
        
        print("Note: DINOv3 is not yet officially released. Using DINOv2 as fallback.")
        
        # Use DINOv2 as fallback
        self.model = DINOv2Classifier(
            model_name='dinov2_vitb14',
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_features(x)
    
    def unfreeze_backbone(self):
        self.model.unfreeze_backbone()
    
    def freeze_backbone(self):
        self.model.freeze_backbone()


def create_dinov2_model(
    config: dict,
    variant: Literal['dinov2', 'dinov2_registers', 'dinov3'] = 'dinov2',
    device: str = 'cuda'
) -> nn.Module:
    """
    Create a DINOv2 model from configuration.
    
    Args:
        config: Configuration dictionary
        variant: Model variant ('dinov2', 'dinov2_registers', or 'dinov3')
        device: Device to place the model on
    
    Returns:
        DINOv2-based model
    """
    if variant == 'dinov2':
        model_config = config['models']['dinov2']
        model = DINOv2Classifier(
            model_name=model_config['name'],
            num_classes=config['dataset']['num_classes'],
            pretrained=model_config['pretrained'],
            freeze_backbone=model_config.get('freeze_backbone', False)
        )
    elif variant == 'dinov2_registers':
        model_config = config['models']['dinov2_registers']
        model = DINOv2RegistersClassifier(
            model_name=model_config['name'],
            num_classes=config['dataset']['num_classes'],
            pretrained=model_config['pretrained'],
            freeze_backbone=model_config.get('freeze_backbone', False)
        )
    elif variant == 'dinov3':
        model_config = config['models'].get('dinov3', config['models']['dinov2'])
        model = DINOv3Classifier(
            model_name=model_config['name'],
            num_classes=config['dataset']['num_classes'],
            pretrained=model_config['pretrained'],
            freeze_backbone=model_config.get('freeze_backbone', False)
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Test the models
    print("Testing DINOv2 Classifier...")
    model = DINOv2Classifier(
        model_name='dinov2_vitb14',
        num_classes=14,
        pretrained=False
    )
    
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    print("\nTesting DINOv2 with Registers...")
    model_reg = DINOv2RegistersClassifier(
        model_name='dinov2_vitb14',
        num_classes=14,
        pretrained=False
    )
    logits_reg = model_reg(x)
    print(f"Output shape (registers): {logits_reg.shape}")
    
    print("\nDINOv2 model tests completed successfully!")
