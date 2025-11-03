# DEQ-MD
Repository for the paper: Deep Equilibrium models for Poisson inverse problems via Mirror Descent  
https://arxiv.org/abs/2507.11461

## Overview

This repository contains the implementation of DEQ-MD, used for image deblurring based on Deep Equilibrium Models (DEQ) with Mirror Descent associated optimization algorithm.

## Main Contents

- `python_files/` – main source code:
  - `train.py` – script for training DEQ models (arguments: `--kernel_type`, `--noise_level`, `--model`).
  - `inference.py` – script for testing/reconstruction on images.
  - `DEQ.py`, `DEQ_utils.py`, `f_theta_2.py` – DEQ implementation and auxiliary modules.
  - `DNCNN.py`, `ICNN.py`– network architectures
  - `Utils.py` – utility functions for image loading, noise addition, etc.
  - `weights_GSDnCNN_denoiser_depth_5.pth` – pre-trained weights for the DnCNN denoiser (used in RED model).
  - `training_images/` – example dataset (`train`, `val`, `test` folders).
  - `training_results/` – automatically generated folder for saving weights and metrics after training.

## Requirements

See `requirements.txt` in the repository root. Main dependencies:
- Python 3.9+ or 3.10+
- PyTorch
- numpy, Pillow, torchvision
- deepinv (for PSNR metric; if unavailable, you can replace with your own implementation)

## Quick Installation

```bash
conda create -n DEQ-MD python=3.11
conda activate DEQ-MD
conda install pip
pip install -r requirements.txt
```

## Usage

### Training
Run from the `DEQs/python_files` folder (or adapt the path):
```bash
python train.py --kernel_type Gauss --noise_level medium --model DEQ-RED
```
- `--kernel_type`: 'Gauss', 'Motion_7', 'Uniform' (required)
- `--noise_level`: 'high', 'medium', 'low' (required)
- `--model`: 'DEQ-RED', 'DEQ-S' (required)

You can add your own kernel types by editing the code.

### Testing / Inference
```bash
python inference.py --kernel Gauss --noise_level medium --regularisation RED
```
Optional flags:
- `--plot_metrics`: plot performance metrics per iteration
- `--save_images`: save ground truth and reconstructed images
- `--save_results`: store average results in a text file

## Output
- Best weights are saved in `DEQs/python_files/training_results` as `weights_{MODEL}_{KERNEL}_{NOISE}.pth`.
- Training and validation metrics are saved as `training_metrics_{MODEL}_{KERNEL}_{NOISE}.pth`.

## Tips
- Make sure you have a GPU and that PyTorch is installed with CUDA support if available. The script automatically selects `cuda` if present.
- For the `DEQ-RED` model, the pre-trained DnCNN denoiser is loaded from `weights_GSDnCNN_denoiser_depth_5.pth` in the `python_files` folder.
- For quick tests, use subsets of data in `training_images/test`.

## Recommended Dataset Structure
- `training_images/train` – training images
- `training_images/val` – validation images
- `training_images/test` – test/inference images

## Useful Files
- `requirements.txt` – Python dependencies
- `train.py` – training
- `inference.py` – inference
- `DEQ_utils.py` – physical operators (blur operator) and helpers

## License & Contact
Use the code respecting the licenses of external dependencies and provided weights. For questions or issues, open an issue in the repository or contact the project author.

## Final Notes
This README is intended as a quick start guide. For advanced modifications (e.g., optimizer, scheduler, architecture), refer directly to the scripts in `python_files`.
