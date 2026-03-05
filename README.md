# Remote-sensing-image-rotated-object-detection-based-on-lightweight-enhanced-detection-head
Remote sensing image rotated object detection based on lightweight enhanced detection head
# Rotated Object Detection with Lightweight Enhanced Detection Head

## Overview
This repository contains the official implementation of the paper: **"Remote sensing image rotated object detection based on lightweight enhanced detection head"**. The code provides a complete framework for rotated object detection in remote sensing images with three key innovations: Multi-layer Feature Interaction Pyramid (MFIP), Lightweight Enhanced Detection Head (LEDH), and KLD loss function.

## Features
= **MFIP (Multi-layer Feature Interaction Pyramid)**: Cross-scale feature fusion beyond adjacent layers
- **LEDH (Lightweight Enhanced Detection Head)**: Parameter-sharing with central difference convolution
- **KLD Loss**: Kullback-Leibler Divergence loss for boundary discontinuity problem
- **LAMP Pruning**: Layer-Adaptive Magnitude-based Pruning for model compression
- **DOTA & HRSC2016 Support**: Full support for both datasets

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+
- mmcv-full 1.5.0+

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/rotated-detection.git
cd rotated-detection

# Create conda environment
conda create -n rotatedet python=3.8
conda activate rotatedet

# Install dependencies
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip install -r requirements.txt
```

## Dataset Preparation

### DOTA Dataset
1. Download DOTA dataset from [official website](https://captain-whu.github.io/DOTA/dataset.html)
2. Organize the dataset as follows:
```
data/
  dota/
    train/
      images/
      labelTxt/
    val/
      images/
      labelTxt/
    test/
      images/
      labelTxt/
```

3. Split images into patches (1024×1024 with 200 overlap):
```bash
python tools/split_dota.py --src data/dota --dst data/dota/split --crop-size 1024 --gap 200
```

### HRSC2016 Dataset
1. Download HRSC2016 dataset
2. Organize as:
```
data/
  hrsc2016/
    train/
      images/
      labelTxt/
    val/
      images/
      labelTxt/
    test/
      images/
      labelTxt/
```

## Model Configuration

Edit `config.py` to adjust training parameters:

```python
# config.py example
TRAIN = {
    'batch_size': 4,           # Batch size per GPU
    'epochs': 100,              # Total training epochs
    'learning_rate': 0.01,      # Initial learning rate
    'momentum': 0.937,          # SGD momentum
    'weight_decay': 0.0005,     # Weight decay
}

DATASET = {
    'dota': {
        'image_size': 1024,
        'num_classes': 15,
    },
    'hrsc2016': {
        'image_size': 800,
        'num_classes': 1,
    }
}
```

## Training

### Train on DOTA
```bash
# Single GPU training
python train.py --dataset dota --config configs/dota.py

# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset dota --config configs/dota.py
```

### Train on HRSC2016
```bash
python train.py --dataset hrsc2016 --config configs/hrsc2016.py
```

### Resume training from checkpoint
```bash
python train.py --resume checkpoints/latest.pth
```

## Evaluation

### Test on DOTA
```bash
python test.py --checkpoint checkpoints/best.pth --dataset dota --out results/dota/
```

### Test on HRSC2016
```bash
python test.py --checkpoint checkpoints/best.pth --dataset hrsc2016 --out results/hrsc2016/
```

### Compute mAP
```bash
python tools/eval.py --pred results/dota/ --gt data/dota/annotations/
```

## Model Pruning (LAMP)

Apply LAMP pruning to reduce model size:
```bash
# Prune trained model
python prune.py --checkpoint checkpoints/best.pth --sparsity 0.3 --out checkpoints/pruned.pth

# Train with pruning
python train.py --prune --target-sparsity 0.3
```

## Results

| Dataset | Method | mAP/AP | Params (M) | GFLOPs |
|---------|--------|--------|------------|--------|
| DOTA | Baseline | 77.04% | 3.0 | 8.3 |
| DOTA | +MFIP | 78.33% | 3.3 | 9.4 |
| DOTA | +LEDH | 79.26% | 2.8 | 7.6 |
| DOTA | +KLD | 80.19% | 2.8 | 7.6 |
| DOTA | +LAMP | **79.54%** | **2.4** | **5.1** |
| HRSC2016 | Ours | **93.04%** | - | - |

## Model Zoo

| Model | Dataset | mAP | Download |
|-------|---------|-----|----------|
| LEDH-R50-DOTA | DOTA | 79.54% | [link]() |
| LEDH-R50-HRSC | HRSC2016 | 93.04% | [link]() |
| LEDH-R50-Pruned | DOTA | 79.54% | [link]() |

## Visualization

Generate detection visualizations:
```bash
python demo.py --image path/to/image.jpg --checkpoint checkpoints/best.pth --out-dir vis/
```

## Project Structure

```
rotated-detection/
├── configs/               # Configuration files
│   ├── dota.py
│   └── hrsc2016.py
├── models/                # Model definitions
│   ├── mfip.py           # Multi-layer Feature Interaction Pyramid
│   ├── ledh.py           # Lightweight Enhanced Detection Head
│   └── detector.py       # Main detector
├── losses/                # Loss functions
│   └── kld_loss.py       # KLD loss
├── pruning/               # Pruning utilities
│   └── lamp.py           # LAMP pruning
├── tools/                 # Utility scripts
│   ├── split_dota.py
│   └── eval.py
├── train.py               # Training script
├── test.py                # Testing script
├── prune.py               # Pruning script
└── demo.py                # Visualization demo
```

## Citation

If you find this code useful for your research, please cite our paper:

```
@article{zhang2025remote,
  title={Remote sensing image rotated object detection based on lightweight enhanced detection head},
  author={Zhang, Luqi and Zhang, Yunzuo and Liu, Ting and Li, Yingxu and Wang, Kai and Sun, YuChuan and Tao, Ran},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please contact:
- Luqi Zhang: zhangluqi678@sina.com
- Yunzuo Zhang: zhangyunzuo888@sina.com

## Acknowledgments

- Thanks to the providers of DOTA and HRSC2016 datasets
- This work was supported by the National Natural Science Foundation of China (No. 61702347)
