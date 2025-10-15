# 🏥 Medical X-Ray Analysis and Report Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A deep learning system for automated chest X-ray analysis using Vision Transformers, with AI-powered radiology report generation via Google Gemini.

## 🌟 Features

- **Multiple Vision Transformer Models**: ViT, Swin V2, DINOv2, DINOv2 with Registers
- **Multi-Label Classification**: Detects 14 different chest pathologies simultaneously
- **Class Imbalance Handling**: Automatic weighted loss for imbalanced medical datasets
- **Distributed Training**: Multi-GPU support with PyTorch DDP
- **Mixed Precision Training**: Faster training with automatic mixed precision (AMP)
- **AI Report Generation**: Automated radiology reports using Google Gemini 2.5 Pro
- **Comprehensive Metrics**: AUC, F1, Precision, Recall, and per-class performance tracking
- **TensorBoard Integration**: Real-time training visualization

## 📊 Dataset

Trained on the **CheXpert dataset** - a large public dataset for chest radiograph interpretation:
- **191,027 training images**
- **202 validation images**
- **14 pathology labels**: No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices

### Dataset Download

1. Visit [CheXpert-v1.0-small](https://www.kaggle.com/datasets/ashery/chexpert/data)
2. download the dataset
3. Extract to your desired location

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# CUDA-capable GPU (recommended: NVIDIA A100, V100, or RTX 3090+)
nvidia-smi
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/xray-analysis.git
cd xray-analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your API keys

# Google Gemini API Configuration
#GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Weights & Biases
#WANDB_API_KEY=your_wandb_api_key_here

# Data paths
#CHEXPERT_DATA_PATH=/path/to/chexpert/data

```

4. **Configure paths**

Edit `config.yaml`:
```yaml
dataset:
  data_dir: "/path/to/your/chexpert/data"
  csv_path: "/path/to/train.csv"
  val_csv_path: "/path/to/valid.csv"
```

### Training

#### Single Model Training

```bash
# Train Vision Transformer (224x224)
python train.py --model vit

# Train Swin Transformer V2 (192x192)
python train.py --model swin_v2

# Train DINOv2 (518x518) - reduce batch size for memory
python train.py --model dinov2

# Train DINOv2 with Registers
python train.py --model dinov2_registers
```

#### Multi-GPU Distributed Training

```bash
# Train on 3 GPUs
python train_distributed.py --model vit

# Or train all models sequentially
python train_distributed.py --model all
```

#### Custom Configuration

```bash
# Custom output directory
python train.py --model vit --output_dir ./my_checkpoints

# Resume from checkpoint
python train.py --model vit --resume ./checkpoints/vit/vit_best.pth

# Specify config file
python train.py --config my_config.yaml --model vit
```

### Inference

```bash
# Single image inference
python inference.py --image path/to/xray.jpg --model vit --checkpoint ./checkpoints/vit/vit_best.pth

# Batch inference with ensemble
python inference.py --image_dir path/to/xrays/ --ensemble --output_dir ./predictions
```

### Evaluation

```bash
# Evaluate model on validation set
python evaluate.py --model vit --checkpoint ./checkpoints/vit/vit_best.pth
```

## 📁 Project Structure

```
xray-analysis/
├── models/
│   ├── __init__.py
│   ├── vit_model.py          # Vision Transformer implementation
│   ├── swin_model.py         # Swin Transformer V2 implementation
│   ├── dinov2_model.py       # DINOv2 implementation
│   └── ensemble.py           # Model ensemble utilities
├── utils/
│   ├── __init__.py
│   ├── dataset.py            # Dataset loading and augmentation
│   ├── metrics.py            # Evaluation metrics
│   └── visualization.py      # Plotting and visualization
├── gemini/
│   ├── __init__.py
│   └── report_generator.py  # AI report generation
├── train.py                  # Single-GPU training script
├── train_distributed.py      # Multi-GPU training script
├── inference.py              # Inference script
├── evaluate.py               # Evaluation script
├── config.yaml               # Main configuration file
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```

## 🎯 Model Performance

### Expected Results (after fixes)

| Model | Image Size | AUC | F1-Score | Parameters |
|-------|-----------|-----|----------|------------|
| ViT Base | 224×224 | 0.85-0.88 | 0.35-0.42 | 86M |
| Swin V2 Base | 192×192 | 0.86-0.89 | 0.36-0.43 | 88M |
| DINOv2 Base | 518×518 | 0.87-0.91 | 0.38-0.45 | 86M |
| DINOv2 + Registers | 518×518 | 0.88-0.92 | 0.39-0.46 | 86M |

*Note: Results vary based on hardware, batch size, and training time*

## ⚙️ Configuration

### Key Configuration Options

**Model-Specific Image Sizes:**
```yaml
models:
  vit:
    image_size: 224      # Native size for ViT
  swin_v2:
    image_size: 192      # Native size for Swin V2
  dinov2:
    image_size: 518      # Native size for DINOv2
```

**Training Parameters:**
```yaml
training:
  batch_size: 128         # Reduce for DINOv2 (32-64)
  num_epochs: 50
  learning_rate: 0.0001   # Lower LR for medical images
  weight_decay: 0.01
  mixed_precision: true   # Enable AMP
```

**Data Augmentation:**
```yaml
augmentation:
  train:
    horizontal_flip: 0.5
    rotation_limit: 10
    brightness_contrast: 0.2
```

## 🔬 Key Features Explained

### 1. Class Imbalance Handling

Medical datasets are highly imbalanced. This system automatically calculates class weights:

```python
# Rare diseases get higher weight
pos_weight = negative_samples / positive_samples

# Example:
# Pneumonia (2.7% positive): weight = 36.1
# Support Devices (51.9% positive): weight = 0.9
```

### 2. Uncertain Label Handling

CheXpert contains uncertain labels (-1). This system uses the **U-Zeros** approach:
- Uncertain → Negative (conservative)
- Reduces training noise
- Standard practice for medical imaging

### 3. Multi-Label Classification

Supports simultaneous detection of multiple pathologies:
- Uses BCEWithLogitsLoss (binary cross-entropy per label)
- Independent prediction per disease
- Reflects real-world radiology scenarios

### 4. Proper Metrics

**Primary Metrics:**
- **ROC AUC** (macro/micro): Main performance indicator
- **F1-Score**: Balance of precision and recall
- **Per-class AUC**: Individual disease performance

**Note**: Standard accuracy is NOT appropriate for multi-label classification.

## 🖥️ Hardware Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 16GB VRAM (e.g., RTX 4090, V100)
- **RAM**: 32GB
- **Storage**: 100GB for dataset + models

### Recommended (for DINOv2 518×518)
- **GPU**: NVIDIA A100 (80GB) or 3× A100 (40GB)
- **RAM**: 128GB+
- **Storage**: 500GB SSD

### Batch Size Guidelines

| Model | Image Size | Batch Size (per GPU) | GPU Memory |
|-------|-----------|---------------------|------------|
| ViT | 224×224 | 128 | 16-24GB |
| Swin V2 | 192×192 | 128 | 14-20GB |
| DINOv2 | 518×518 | 32-64 | 40-80GB |
| DINOv2 | 224×224 | 128 | 16-24GB |

## 📈 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir runs/
```

View at: http://localhost:6006

**Logged Metrics:**
- Training/Validation Loss
- AUC (macro/micro)
- F1, Precision, Recall
- Learning Rate
- Per-class performance

### Checkpoints

Checkpoints are saved in `checkpoints/{model_name}/`:
- `{model_name}_best.pth`: Best validation AUC
- `{model_name}_epoch_{N}.pth`: Periodic checkpoints
- Training history and metrics included

## 🤖 AI Report Generation

Generate professional radiology reports using Google Gemini:

```python
from gemini.report_generator import ReportGenerator

# Initialize generator
generator = ReportGenerator(api_key="your_api_key")

# Generate report
predictions = {
    'Cardiomegaly': 0.85,
    'Pleural Effusion': 0.72,
    'Support Devices': 0.91
}

report = generator.generate_report(
    image_path="xray.jpg",
    predictions=predictions
)

print(report)
```

**Example Output:**
```
CHEST RADIOGRAPH REPORT

CLINICAL INDICATION:
Evaluation of cardiac size and pulmonary findings.

FINDINGS:
- Cardiomegaly present with high confidence (85%)
- Bilateral pleural effusion noted (72%)
- Support devices in place, properly positioned (91%)
- No evidence of pneumothorax
- Lung fields otherwise clear

IMPRESSION:
Cardiomegaly with bilateral pleural effusions. Support devices in appropriate position.

RECOMMENDATIONS:
- Clinical correlation recommended
- Consider echocardiography for cardiac assessment
- Follow-up imaging to assess pleural effusion progression
```

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors

```bash
# Reduce batch size
python train.py --model dinov2 --batch_size 32

# Or use smaller image size (edit config.yaml)
image_size: 224  # instead of 518
```

### Poor Performance

See `PERFORMANCE_FIXES.md` for detailed optimization guide:
- Check class imbalance handling
- Verify uncertain label policy
- Adjust learning rate
- Monitor per-class metrics

### CUDA Errors

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Set specific GPU
export CUDA_VISIBLE_DEVICES=0,1,2
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


**Disclaimer**: This tool is for research purposes only and should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.
