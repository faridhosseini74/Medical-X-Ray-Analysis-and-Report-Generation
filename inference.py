#!/usr/bin/env python3
# Inference script for X-ray analysis and report generation

import os
import argparse
import yaml
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import json
from datetime import datetime

from models import ViTClassifier, SwinV2Classifier, DINOv2Classifier, DINOv2RegistersClassifier
from models.dinov2_model import DINOv3Classifier
from models.ensemble import EnsembleModel
from utils.dataset import get_transforms
from gemini.report_generator import GeminiReportGenerator
from utils.visualization import visualize_predictions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Inference for X-ray classification and report generation')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--image', type=str, required=False,
                        help='Path to single X-ray image')
    parser.add_argument('--input_dir', type=str, required=False,
                        help='Directory containing X-ray images')
    parser.add_argument('--output_dir', type=str, default='outputs/reports',
                        help='Output directory for reports')
    parser.add_argument('--model', type=str, default='vit',
                        choices=['vit', 'swin_v2', 'dinov2', 'dinov2_registers', 'dinov3'],
                        help='Model to use for inference')
    parser.add_argument('--checkpoint', type=str, required=False,
                        help='Path to model checkpoint')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use ensemble of models')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--generate_report', action='store_true', default=True,
                        help='Generate radiological report using Gemini')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generate visualization of predictions')
    return parser.parse_args()


class InferencePipeline:
    """Pipeline for X-ray inference and report generation."""
    
    def __init__(self, config, model, device='cuda', report_generator=None):
        """
        Initialize inference pipeline.
        
        Args:
            config: Configuration dictionary
            model: Trained model
            device: Device to run on
            report_generator: GeminiReportGenerator instance
        """
        self.config = config
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.report_generator = report_generator
        
        # Get transforms
        self.transform = get_transforms(config, mode='val')
        
        # Labels
        self.labels = config['dataset']['labels']
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for inference.
        
        Args:
            image_path: Path to the image
        
        Returns:
            Preprocessed image tensor
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        return image_tensor.unsqueeze(0)  # Add batch dimension
    
    def predict(self, image_tensor):
        """
        Run model prediction.
        
        Args:
            image_tensor: Preprocessed image tensor
        
        Returns:
            Prediction probabilities
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            probabilities = torch.sigmoid(outputs)
        
        return probabilities.cpu().numpy()[0]
    
    def format_predictions(self, predictions, threshold=0.5):
        """
        Format predictions into a readable dictionary.
        
        Args:
            predictions: Prediction array
            threshold: Threshold for positive classification
        
        Returns:
            Dictionary of predictions
        """
        results = {}
        
        for i, label in enumerate(self.labels):
            score = float(predictions[i])
            results[label] = {
                'score': score,
                'positive': score >= threshold
            }
        
        return results
    
    def run_inference(self, image_path, threshold=0.5, generate_report=True, save_dir=None):
        """
        Run complete inference pipeline on a single image.
        
        Args:
            image_path: Path to X-ray image
            threshold: Classification threshold
            generate_report: Whether to generate a report
            save_dir: Directory to save outputs
        
        Returns:
            Dictionary containing predictions and report
        """
        print(f"\nProcessing: {image_path}")
        
        # Preprocess
        image_tensor = self.preprocess_image(image_path)
        
        # Predict
        predictions = self.predict(image_tensor)
        formatted_predictions = self.format_predictions(predictions, threshold)
        
        # Print predictions
        print("\nPredictions:")
        positive_findings = []
        for label, result in formatted_predictions.items():
            if result['positive']:
                print(f"  ✓ {label}: {result['score']:.3f}")
                positive_findings.append(label)
            elif result['score'] > 0.3:  # Show borderline cases
                print(f"  ? {label}: {result['score']:.3f} (below threshold)")
        
        if not positive_findings:
            print("  No significant findings detected")
        
        # Generate report
        report = None
        if generate_report and self.report_generator:
            print("\nGenerating radiological report...")
            report = self.report_generator.generate_report(
                predictions=predictions,
                image_path=image_path,
                threshold=threshold,
                include_image=True
            )
            
            if report['status'] == 'success':
                print("\n" + "="*80)
                print("RADIOLOGICAL REPORT")
                print("="*80)
                print(report['full_report'])
                print("="*80)
        
        # Save results
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save predictions
            image_name = Path(image_path).stem
            pred_path = os.path.join(save_dir, f'{image_name}_predictions.json')
            with open(pred_path, 'w') as f:
                json.dump(formatted_predictions, f, indent=2)
            
            # Save report
            if report and report['status'] == 'success':
                report_path = os.path.join(save_dir, f'{image_name}_report.txt')
                with open(report_path, 'w') as f:
                    f.write(report['full_report'])
                
                # Save report JSON
                report_json_path = os.path.join(save_dir, f'{image_name}_report.json')
                with open(report_json_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            
            print(f"\n✓ Results saved to: {save_dir}")
        
        return {
            'image_path': image_path,
            'predictions': formatted_predictions,
            'predictions_raw': predictions,
            'report': report,
            'positive_findings': positive_findings
        }
    
    def batch_inference(self, image_paths, threshold=0.5, generate_report=True, save_dir=None):
        """
        Run inference on multiple images.
        
        Args:
            image_paths: List of image paths
            threshold: Classification threshold
            generate_report: Whether to generate reports
            save_dir: Directory to save outputs
        
        Returns:
            List of results dictionaries
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"\n{'='*80}")
            print(f"Image {i+1}/{len(image_paths)}")
            print(f"{'='*80}")
            
            result = self.run_inference(
                image_path=image_path,
                threshold=threshold,
                generate_report=generate_report,
                save_dir=save_dir
            )
            results.append(result)
        
        # Save summary
        if save_dir:
            summary_path = os.path.join(save_dir, 'summary.json')
            summary = {
                'timestamp': datetime.now().isoformat(),
                'num_images': len(image_paths),
                'threshold': threshold,
                'results': [
                    {
                        'image': Path(r['image_path']).name,
                        'positive_findings': r['positive_findings']
                    }
                    for r in results
                ]
            }
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
        
        return results


def load_model(model_name, checkpoint_path, config, device='cuda'):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_name: Name of the model
        checkpoint_path: Path to checkpoint
        config: Configuration dictionary
        device: Device to load on
    
    Returns:
        Loaded model
    """
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
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Loaded model from: {checkpoint_path}")
    
    return model


def main():
    """Main inference function."""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize report generator
    report_generator = None
    if args.generate_report:
        try:
            report_generator = GeminiReportGenerator(
                model_name=config['gemini']['model_name'],
                temperature=config['gemini']['temperature'],
                max_tokens=config['gemini']['max_tokens']
            )
            print("✓ Gemini report generator initialized")
        except Exception as e:
            print(f"Warning: Could not initialize Gemini report generator: {e}")
            print("Continuing without report generation...")
    
    # Load model
    if args.ensemble:
        print("Ensemble mode not fully implemented in this script")
        print("Using single model instead")
    
    # Get checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Use best checkpoint from default location
        checkpoint_dir = os.path.join('checkpoints', args.model)
        checkpoint_path = os.path.join(checkpoint_dir, f'{args.model}_best.pth')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Please train the model first or specify --checkpoint"
            )
    
    # Load model
    model = load_model(args.model, checkpoint_path, config, device)
    
    # Create inference pipeline
    pipeline = InferencePipeline(config, model, device, report_generator)
    
    # Get image paths
    if args.image:
        image_paths = [args.image]
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        image_paths = list(input_dir.glob('*.jpg')) + \
                     list(input_dir.glob('*.png')) + \
                     list(input_dir.glob('*.jpeg'))
        image_paths = [str(p) for p in image_paths]
    else:
        raise ValueError("Either --image or --input_dir must be specified")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run inference
    if len(image_paths) == 1:
        result = pipeline.run_inference(
            image_path=image_paths[0],
            threshold=args.threshold,
            generate_report=args.generate_report,
            save_dir=args.output_dir
        )
    else:
        results = pipeline.batch_inference(
            image_paths=image_paths,
            threshold=args.threshold,
            generate_report=args.generate_report,
            save_dir=args.output_dir
        )
    
    print(f"\n{'='*80}")
    print("Inference completed successfully!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
