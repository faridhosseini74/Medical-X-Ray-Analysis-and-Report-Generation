# Vision Transformer (ViT) Model for X-ray Classification

import torch
import torch.nn as nn
import timm
from typing import Optional, Tuple


class ViTClassifier(nn.Module):
    """
    Vision Transformer (ViT) model for multi-label chest X-ray classification.
    Uses pre-trained ViT from timm library.
    """
    
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        num_classes: int = 14,
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        """
        Args:
            model_name: Name of the ViT model from timm
            num_classes: Number of output classes
            pretrained: Whether to use pre-trained weights
            dropout: Dropout rate before the final classifier
            freeze_backbone: Whether to freeze the backbone during training
        """
        super(ViTClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained ViT model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove the classifier head
            global_pool='avg'  # Use average pooling
        )
        
        # Get the feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim // 2, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize the classifier layers."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
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
        return features
    
    def get_attention_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Extract attention maps from the transformer blocks.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            Tuple of (logits, attention_maps)
        """
        # This is a simplified version; actual implementation may vary
        # depending on the specific ViT architecture
        attention_maps = []
        
        # Forward pass with hooks to capture attention
        def hook_fn(module, input, output):
            if hasattr(output, 'attn'):
                attention_maps.append(output.attn)
        
        hooks = []
        for block in self.backbone.blocks:
            hook = block.attn.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        logits = self.forward(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return logits, attention_maps
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_backbone(self):
        """Freeze the backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False


def create_vit_model(config: dict, device: str = 'cuda') -> ViTClassifier:
    """
    Create a ViT model from configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to place the model on
    
    Returns:
        ViTClassifier model
    """
    model_config = config['models']['vit']
    
    model = ViTClassifier(
        model_name=model_config['name'],
        num_classes=config['dataset']['num_classes'],
        pretrained=model_config['pretrained'],
        dropout=model_config['dropout']
    )
    
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Test the model
    model = ViTClassifier(
        model_name='vit_base_patch16_224',
        num_classes=14,
        pretrained=False
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test feature extraction
    features = model.get_features(x)
    print(f"Feature shape: {features.shape}")
    
    print("ViT model test completed successfully!")
