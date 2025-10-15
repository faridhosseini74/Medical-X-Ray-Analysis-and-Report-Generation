# Ensemble Model for combining multiple vision transformers

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np


class EnsembleModel(nn.Module):
    """
    Ensemble model that combines predictions from multiple vision transformers.
    Supports different aggregation strategies.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        model_names: List[str],
        aggregation: str = 'mean',
        weights: Optional[List[float]] = None
    ):
        """
        Args:
            models: List of model instances
            model_names: Names of the models for identification
            aggregation: Aggregation strategy ('mean', 'weighted', 'max', 'voting')
            weights: Weights for weighted aggregation (must sum to 1)
        """
        super(EnsembleModel, self).__init__()
        
        self.model_names = model_names
        self.aggregation = aggregation
        
        # Store models as ModuleList
        self.models = nn.ModuleList(models)
        
        # Set up weights
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
            self.weights = weights
        
        print(f"Ensemble created with {len(models)} models:")
        for name, weight in zip(model_names, self.weights):
            print(f"  - {name}: weight={weight:.3f}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all models and aggregate predictions.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            Aggregated logits of shape (batch_size, num_classes)
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            with torch.set_grad_enabled(self.training):
                pred = model(x)
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (num_models, batch_size, num_classes)
        
        # Aggregate
        if self.aggregation == 'mean':
            output = predictions.mean(dim=0)
        
        elif self.aggregation == 'weighted':
            weights = torch.tensor(self.weights, device=predictions.device, dtype=predictions.dtype)
            weights = weights.view(-1, 1, 1)  # (num_models, 1, 1)
            output = (predictions * weights).sum(dim=0)
        
        elif self.aggregation == 'max':
            output, _ = predictions.max(dim=0)
        
        elif self.aggregation == 'voting':
            # Apply sigmoid and threshold at 0.5 for voting
            binary_preds = (torch.sigmoid(predictions) > 0.5).float()
            output = binary_preds.mean(dim=0)
            # Convert back to logits (approximate)
            output = torch.logit(output.clamp(min=1e-7, max=1-1e-7))
        
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation}")
        
        return output
    
    def predict_with_individual_scores(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get predictions from individual models as well as the ensemble.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            Dictionary with keys 'ensemble' and individual model names
        """
        results = {}
        
        # Get individual predictions
        for name, model in zip(self.model_names, self.models):
            with torch.no_grad():
                pred = model(x)
                results[name] = torch.sigmoid(pred)
        
        # Get ensemble prediction
        with torch.no_grad():
            ensemble_pred = self.forward(x)
            results['ensemble'] = torch.sigmoid(ensemble_pred)
        
        return results
    
    def get_model_contributions(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Analyze the contribution of each model to the ensemble prediction.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            Dictionary with model contributions
        """
        contributions = {}
        
        # Get predictions
        predictions = []
        for name, model in zip(self.model_names, self.models):
            with torch.no_grad():
                pred = torch.sigmoid(model(x))
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        # Calculate variance contribution
        pred_mean = predictions.mean(dim=0)
        
        for i, name in enumerate(self.model_names):
            # Calculate how much each model differs from the mean
            diff = torch.abs(predictions[i] - pred_mean).mean().item()
            contributions[name] = {
                'variance': diff,
                'weight': self.weights[i]
            }
        
        return contributions
    
    def eval_individual(self):
        """Set all models to eval mode."""
        for model in self.models:
            model.eval()
    
    def train_individual(self, mode: bool = True):
        """Set all models to train mode."""
        for model in self.models:
            model.train(mode)


def create_ensemble_from_checkpoints(
    model_classes: List[type],
    checkpoint_paths: List[str],
    model_names: List[str],
    config: dict,
    device: str = 'cuda',
    aggregation: str = 'mean',
    weights: Optional[List[float]] = None
) -> EnsembleModel:
    """
    Create an ensemble model from saved checkpoints.
    
    Args:
        model_classes: List of model class constructors
        checkpoint_paths: List of paths to model checkpoints
        model_names: Names for the models
        config: Configuration dictionary
        device: Device to load models on
        aggregation: Aggregation strategy
        weights: Optional weights for weighted aggregation
    
    Returns:
        EnsembleModel instance
    """
    models = []
    
    for model_class, checkpoint_path, name in zip(model_classes, checkpoint_paths, model_names):
        print(f"Loading {name} from {checkpoint_path}...")
        
        # Create model instance
        if name.startswith('vit'):
            from .vit_model import create_vit_model
            model = create_vit_model(config, device)
        elif name.startswith('swin'):
            from .swin_model import create_swin_model
            model = create_swin_model(config, device)
        elif name.startswith('dino'):
            from .dinov2_model import create_dinov2_model
            variant = 'dinov2_registers' if 'reg' in name else 'dinov2'
            model = create_dinov2_model(config, variant, device)
        else:
            raise ValueError(f"Unknown model type: {name}")
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        models.append(model)
    
    # Create ensemble
    ensemble = EnsembleModel(
        models=models,
        model_names=model_names,
        aggregation=aggregation,
        weights=weights
    )
    
    ensemble = ensemble.to(device)
    ensemble.eval()
    
    return ensemble


if __name__ == "__main__":
    # Test ensemble
    from .vit_model import ViTClassifier
    from .swin_model import SwinV2Classifier
    
    print("Creating test models...")
    model1 = ViTClassifier(
        model_name='vit_base_patch16_224',
        num_classes=14,
        pretrained=False
    )
    
    model2 = SwinV2Classifier(
        model_name='swinv2_tiny_window8_256',
        num_classes=14,
        pretrained=False
    )
    
    print("\nCreating ensemble...")
    ensemble = EnsembleModel(
        models=[model1, model2],
        model_names=['vit', 'swin'],
        aggregation='mean'
    )
    
    print("\nTesting forward pass...")
    x = torch.randn(2, 3, 224, 224)
    output = ensemble(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\nTesting individual predictions...")
    results = ensemble.predict_with_individual_scores(x)
    for name, pred in results.items():
        print(f"{name}: {pred.shape}")
    
    print("\nEnsemble model test completed successfully!")
