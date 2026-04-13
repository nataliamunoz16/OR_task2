# Fashion segmentation for fashionpedia

This repository contains a PyTorch-based semantic segmentation pipeline for fashion images, using Fashionpedia-style training and validation data. It includes dataset loaders, multiple segmentation architectures, training/evaluation loops, class balancing utilities and visualization helpers.

## Project Overview

- `main.py`: Primary training, validation and test evaluation script for segmentation models.
- `4_channels_train.py`, `detection.py`: Detection + segmentation training pipeline for a 4-channel model variant that adds bounding-box heatmap information.
- `dataset.py`: Dataset classes for full images, cropped instance patches and bounding-box-enhanced data.
- `utils.py`: DataLoader utilities, training/evaluation loops, metrics, plotting and prediction visualization.
- `config.py`: Dataset and output path configuration.
- `overrepresented_classes.py`: Utility for selecting overrepresented classes and fine-tuning training.
- `Segformer.py`, `deeplabv3plus.py`, `unet.py`: Model definitions and wrappers for segmentation backbones.
- `YOLO.py`: Training, validation and testing for YOLO instance segmentation model.
- `fashionpedia_all.py`, `augment_fashionpedia.py`: Additional data preparation and augmentation scripts.

## Key Features

- Semantic segmentation on fashion images with up to 27 apparel/accessory classes plus background.
- Support for multiple architectures: DeepLabV3+, SegFormer and U-Net.
- Support YOLO architecture.
- Train/validation/test split creation and evaluation metrics including mIoU, mean Dice and per-class scores.
- Visualization of predicted segmentation masks vs ground truth.
- Optional overrepresented-class filtering and fine-tuning support.

## Requirements

Recommended Python packages:

- Python 3.8+
- torch
- torchvision
- segmentation_models_pytorch
- pillow
- numpy
- pandas
- matplotlib
- tqdm
- pycocotools
- timm
- einops

Install dependencies with pip, for example:

```bash
pip install torch torchvision segmentation-models-pytorch pillow numpy pandas matplotlib tqdm pycocotools timm einops
```

## Setup

1. Place the dataset files in the directories referenced by `config.py`.
2. Update the paths in `config.py` to match your local filesystem if needed.
3. Confirm that the following files/directories exist:
   - `train2020/train/`
   - `val_test2020/test/`
   - `mask_train/`
   - `mask_val/`

## Running Training

### Standard segmentation

Run the main training pipeline:

```bash
python main.py
```

This will:

- load train/validation/test datasets
- build a model based on the chosen `model_name`
- train the model
- save best checkpoints to `models/`
- evaluate the best models on the test set
- save test metrics to `results/`
- generate prediction visualizations

### 4-channel training

Run the 4-channel model pipeline:

```bash
python 4_channels_train.py
```

This script builds a model variant that expects an extra channel of bounding-box-derived heatmap features.

## Configuration

The repository uses `config.py` for dataset and output paths. Example path variables:

- `ROOT`: root data path
- `TRAIN_IMG`, `TEST_IMG`: image directories
- `TRAIN_MASK`, `TEST_MASK`: segmentation mask directories
- `MODELS`: checkpoint output directory
- `RESULTS`: metrics and plots output directory
- `ANNOTATIONS_TRAIN`, `ANNOTATIONS_TEST`: COCO-style annotation JSON files

## Dataset Format

The dataset loaders expect:

- training images as `.jpg`
- segmentation masks as `_seg.png`
- matching file basenames between images and masks
- optional COCO annotation JSON files for cropped samples and fine-tuning

## Output

The training scripts save:

- model checkpoints in `models/`
- validation metrics JSON in `results/`
- test metrics JSON in `results/`
- visualization plots as PNG files in `results/`

## Notes

- `main.py` supports optional fine-tuning behavior via overrepresented class filtering and uses a separate `best_model_path` checkpoint for this mode.
- The dataset classes resize input images while preserving aspect ratio and pad them to fixed size.
- Loss configuration can use standard cross-entropy or focal loss in `main.py`.
- The repository currently contains Linux-style default paths in `config.py`; update them before running on Windows.

