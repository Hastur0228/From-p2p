# Foot Insole Deformation Network

A deep learning system for generating personalized shoe insoles from foot scans using point cloud deformation networks.

## Overview

This project implements a deformation network that transforms a template insole point cloud into a personalized insole based on foot geometry. The system uses:

- **DGCNN Encoder**: Extracts global features from foot point clouds
- **Deformation Network**: Regresses point-wise deformations to transform template to target
- **Chamfer Distance Loss**: Ensures geometric similarity between predicted and target insoles

## Features

- 🦶 **Foot-to-Insole Mapping**: Direct transformation from foot scans to personalized insoles
- 🔄 **Template-Based Deformation**: Uses average insole templates as starting points
- 📊 **Interactive Training**: Command-line interface with parameter customization
- 🎯 **Early Stopping**: Configurable patience-based training termination
- 📈 **Visualization**: Training curves and metrics tracking
- 🔧 **Data Augmentation**: Point cloud augmentation for improved generalization
- 📁 **Multi-Side Support**: Train separate models for left/right feet
- 🚀 **Optimized Defaults**: Pre-configured with proven hyperparameters for better performance

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd foot-insole-deformation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import trimesh; print('Trimesh installed successfully')"
   ```

## Data Preparation

### Directory Structure

```
data/
├── pointcloud/
│   ├── feet/          # Foot scan point clouds (.npy)
│   └── insoles/       # Target insole point clouds (.npy)
└── raw/               # Original data files
```

### Data Format

- **Input**: Foot point clouds as `.npy` files (N×3 or N×6 arrays)
- **Target**: Insole point clouds as `.npy` files (N×3 arrays)
- **Template**: Average insole template as `.npz` file

### Data Processing

1. **Convert scans to point clouds**
   ```bash
   python scripts/stl2pointcloud_withpreprocess.py --input-dir data/raw --output-dir data/pointcloud
   ```

2. **Generate average templates**
   ```bash
   python scripts/average_insole.py --data-dir data/pointcloud/insoles --output Templates/
   ```

## Usage

### Quick Start (Recommended)

```bash
# Train with optimized default configuration
python train_deformnet.py

# This uses the following optimized settings:
# - 200 epochs with early stopping
# - Plateau scheduler with warmup
# - Enhanced regularization (higher dropout, weight decay)
# - Local Chamfer distance for better detail preservation
# - Data augmentation enabled
```

### Basic Training

```bash
# Train with custom parameters
python train_deformnet.py \
    --data-root data/pointcloud \
    --template Templates/average_insole_template.npz \
    --batch-size 16 \
    --epochs 300 \
    --lr 1e-4
```

### Interactive Training

```bash
# Launch interactive parameter configuration
python train_deformnet.py --no-interactive
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-root` | `data/pointcloud` | Data directory path |
| `--template` | `Templates/average_insole_template.npz` | Template insole path |
| `--batch-size` | `12` | Training batch size |
| `--epochs` | `200` | Maximum training epochs |
| `--lr` | `5e-4` | Learning rate |
| `--weight-decay` | `1e-3` | Weight decay for regularization |
| `--scheduler` | `plateau` | Learning rate scheduler |
| `--warmup-epochs` | `5` | Learning rate warmup epochs |
| `--num-points` | `4096` | Points per sample |
| `--early-stopping` | `False` | Enable early stopping |
| `--patience` | `20` | Early stopping patience |
| `--side` | `LR` | Train L/R/LR feet |
| `--local-cd-weight` | `0.1` | Local Chamfer distance weight |
| `--aug-enable` | `False` | Enable data augmentation |

### Advanced Options

```bash
# Custom learning rate schedule
python train_deformnet.py \
    --scheduler cosine \
    --warmup-epochs 10 \
    --lr 1e-3

# Enhanced regularization
python train_deformnet.py \
    --dgcnn-dropout 0.5 \
    --mlp-dropout 0.5 \
    --weight-decay 2e-3

# Custom model architecture
python train_deformnet.py \
    --dgcnn-feat-dim 512 \
    --hidden-dims "512,256,128" \
    --dgcnn-k 30
```

## Model Architecture

### DGCNN Encoder
- **Input**: Foot point cloud (N×3 or N×6)
- **Output**: Global feature vector (256-dim, optimized)
- **Features**: Multi-scale EdgeConv, global pooling, enhanced dropout (0.3)

### Deformation Network
- **Input**: Template points + Global features
- **Output**: Deformed point cloud
- **Architecture**: MLP with skip connections, enhanced dropout (0.4)

### Loss Functions
- **Global Chamfer Distance**: Overall shape similarity (weight: 1.0)
- **Local Chamfer Distance**: Local geometric details (weight: 0.1, enabled by default)
- **L2 Regularization**: Smooth deformations (weight decay: 1e-3)

## Training Process

1. **Data Loading**: Load foot-insole pairs with augmentation (8x multiplier)
2. **Feature Extraction**: Encode foot geometry using DGCNN (256-dim features)
3. **Deformation**: Transform template using global features
4. **Loss Computation**: Calculate combined Chamfer distance and regularization
5. **Optimization**: Update network parameters with Adam optimizer
6. **Learning Rate Schedule**: Plateau scheduler with 5-epoch warmup
7. **Validation**: Evaluate on validation set
8. **Early Stopping**: Monitor validation loss for convergence

## Optimized Configuration Details

The default configuration has been optimized for better performance:

### Training Strategy
- **Extended Training**: 200 epochs with early stopping
- **Learning Rate**: 5e-4 with plateau scheduler
- **Warmup**: 5 epochs for stable training start
- **Regularization**: Enhanced dropout (0.3/0.4) and weight decay (1e-3)

### Model Architecture
- **Feature Dimension**: Reduced to 256 for efficiency
- **Dropout**: Increased for better generalization
- **Local Loss**: Enabled by default for detail preservation

### Data Augmentation
- **Enabled by Default**: 8x multiplier for robust training
- **Multiple Techniques**: Jitter, dropout patches, resampling

## Output

### Checkpoints
- `checkpoints/L/best.pth`: Best model for left foot
- `checkpoints/R/best.pth`: Best model for right foot
- `checkpoints/L/final.pth`: Final model for left foot

### Logs
- `Log/train/`: Training logs and metrics
- `Log/train/L/metrics.csv`: Training history
- `Log/train/L/loss_curve.png`: Loss visualization

### Model Checkpoint Contents
```python
{
    'epoch': int,           # Training epoch
    'encoder': dict,        # DGCNN encoder state
    'regressor': dict,      # Deformation network state
    'optimizer': dict,      # Optimizer state
    'best_val': float,      # Best validation loss
    'args': dict,           # Training arguments
    'side': str             # L/R foot identifier
}
```

## Evaluation

### Metrics
- **Chamfer Distance**: Geometric similarity measure
- **Training Loss**: Convergence monitoring
- **Validation Loss**: Generalization assessment

### Visualization
```bash
# Plot training curves
python -c "
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('Log/train/L/metrics.csv')
plt.plot(df['epoch'], df['train_loss'], label='Train')
plt.plot(df['epoch'], df['val_loss'], label='Validation')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.savefig('training_curves.png')
"
```

## Performance Comparison

### Default Configuration Benefits
- **Faster Convergence**: Plateau scheduler with warmup
- **Better Generalization**: Enhanced regularization
- **Improved Details**: Local Chamfer distance enabled
- **Robust Training**: Data augmentation by default

### Recommended Settings for Different Scenarios

#### High-Performance Training
```bash
python train_deformnet.py \
    --batch-size 24 \
    --dgcnn-feat-dim 512 \
    --aug-multiplier 16
```

#### Memory-Constrained Training
```bash
python train_deformnet.py \
    --batch-size 8 \
    --num-points 2048 \
    --dgcnn-feat-dim 128
```

#### Fast Prototyping
```bash
python train_deformnet.py \
    --epochs 50 \
    --patience 10 \
    --aug-multiplier 4
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size and feature dimension
   python train_deformnet.py --batch-size 8 --dgcnn-feat-dim 128
   ```

2. **Slow Training**
   ```bash
   # Increase batch size, reduce complexity
   python train_deformnet.py --batch-size 24 --num-points 2048
   ```

3. **Poor Convergence**
   ```bash
   # Try different learning rate and scheduler
   python train_deformnet.py --lr 1e-4 --scheduler cosine
   ```

### Performance Tips

- Use GPU acceleration for faster training
- The default configuration is optimized for most use cases
- Monitor validation loss to prevent overfitting
- Enable data augmentation for better generalization
- Use appropriate batch size for your GPU memory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{foot_insole_deformation,
  title={Foot Insole Deformation Network},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Acknowledgments

- DGCNN architecture from [Point Cloud Learning with DGCNN](https://github.com/WangYueFt/dgcnn)
- Chamfer distance implementation from [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
- Point cloud processing utilities from [Open3D](http://www.open3d.org/)
