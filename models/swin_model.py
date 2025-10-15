# Swin Transformer V2 Model for X-ray Classification

import torch
import torch.nn as nn
import timm
from typing import Optional


class SwinV2Classifier(nn.Module):
    """
    Swin Transformer V2 model for multi-label chest X-ray classification.
    Uses pre-trained Swin V2 from timm library.
    """
    
    def __init__(
        self,
        model_name: str = 'swinv2_base_window12to16_192to256_22kft1k',
        num_classes: int = 14,
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        """
        Args:
            model_name: Name of the Swin V2 model from timm
            num_classes: Number of output classes
            pretrained: Whether to use pre-trained weights
            dropout: Dropout rate before the final classifier
            freeze_backbone: Whether to freeze the backbone during training
        """
        super(SwinV2Classifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained Swin V2 model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove the classifier head
        )
        
        # Get the feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1) if hasattr(self.backbone, 'head') else nn.Identity(),
            nn.Flatten(),
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim // 2, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize the classifier layers."""
        for m in self.classifier.modules():
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
        if len(features.shape) == 3:  # (B, N, C)
            features = features.mean(dim=1)  # Global average pooling
        
        # Classify
        logits = self.classifier(features)
        
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
            if len(features.shape) == 3:
                features = features.mean(dim=1)
        return features
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_backbone(self):
        """Freeze the backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def progressive_unfreeze(self, stage: int = 1):
        """
        Progressively unfreeze layers from the end.
        
        Args:
            stage: Stage of unfreezing (1 = last layer, 2 = last 2 layers, etc.)
        """
        # Freeze all first
        self.freeze_backbone()
        
        # Unfreeze progressively
        if hasattr(self.backbone, 'layers'):
            layers = self.backbone.layers
            num_layers = len(layers)
            
            for i in range(max(0, num_layers - stage), num_layers):
                for param in layers[i].parameters():
                    param.requires_grad = True


def create_swin_model(config: dict, device: str = 'cuda') -> SwinV2Classifier:
    """
    Create a Swin Transformer V2 model from configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to place the model on
    
    Returns:
        SwinV2Classifier model
    """
    model_config = config['models']['swin_v2']
    
    model = SwinV2Classifier(
        model_name=model_config['name'],
        num_classes=config['dataset']['num_classes'],
        pretrained=model_config['pretrained'],
        dropout=model_config['dropout']
    )
    
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Test the model
    model = SwinV2Classifier(
        model_name='swinv2_tiny_window8_256',
        num_classes=14,
        pretrained=False
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test feature extraction
    features = model.get_features(x)
    print(f"Feature shape: {features.shape}")
    
    print("Swin Transformer V2 model test completed successfully!")
