# basic imports
import numpy as np
# DL library imports
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

# libraries for loading image, plotting 
import cv2
import matplotlib.pyplot as plt

from einops import rearrange
import segmentation_models_pytorch as smp
from timm.models.layers import drop_path, trunc_normal_
from utils import get_dataloaders, inverse_transform
from dataset import FashionDataset
import config
from torch.utils.data import random_split
import torchvision.transforms as T
from pathlib import Path
from utils import meanIoU                  # metric class
from utils import plot_training_results    # function to plot training curves
from utils import evaluate_model           # evaluation function
from utils import train_validate_model     # train validate function
from Segformer import segformer_mit_b3
"""
Based on the https://colab.research.google.com/gist/Jeremy26/13f71c273f0a1a93f758d02b2b77802e/segformer-cityscapes-run.ipynb?authuser=0#scrollTo=370dd128
"""

#dataset loader
targetWidth = 512
targetHeight = 768
transform = T.Compose([T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
train_set = FashionDataset(config.TRAIN_IMG, config.TRAIN_MASK, targetHeight, targetWidth, transform)
val_test_set = FashionDataset(config.TEST_IMG, config.TEST_MASK, targetHeight, targetWidth, transform)
val_size = int(0.5 * len(val_test_set))
test_size = len(val_test_set) - val_size
val_set, test_set = random_split(val_test_set, [val_size, test_size])
train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_set, val_set, test_set)


#Training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

# MODEL HYPERPARAMETERS
N_EPOCHS = 30
NUM_CLASSES = 28
MAX_LR = 1e-3
MODEL_NAME = f'segformer_{targetHeight}_{targetWidth}_CE_loss'


# criterion = smp.losses.DiceLoss('multiclass', classes=np.arange(20).tolist(), log_loss = True, smooth=1.0)
criterion = nn.CrossEntropyLoss(ignore_index=19)

# create model, load imagenet pretrained weights
model = segformer_mit_b3(in_channels=3, num_classes=NUM_CLASSES).to(device)
model.backbone.load_state_dict(torch.load('segformers/segformer_mit_b3_imagenet_weights.pt', map_location=device))


# create optimizer, lr_scheduler and pass to training function
optimizer = optim.Adam(model.parameters(), lr=MAX_LR)
scheduler = OneCycleLR(optimizer, max_lr= MAX_LR, epochs = N_EPOCHS,steps_per_epoch = len(train_dataloader), 
                       pct_start=0.3, div_factor=10, anneal_strategy='cos')

_ = train_validate_model(model, N_EPOCHS, MODEL_NAME, criterion, optimizer, 
                         device, train_dataloader, val_dataloader, meanIoU, 'meanIoU',
                         NUM_CLASSES, lr_scheduler = scheduler, output_path = config.ROOT)
model.load_state_dict(torch.load(f'{MODEL_NAME}.pt', map_location=device))
_, test_metric = evaluate_model(model, test_dataloader, criterion, meanIoU, NUM_CLASSES, device)
print(f"\nModel has {test_metric} mean IoU in test set")

id_to_color = np.array([[i,i,i] for i in range(29)], dtype=np.uint8)
id_to_color = np.array([
    [0,0,0],
    [255,0,0],
    [0,255,0],
    [0,0,255],
    [255,255,0],
    [255,0,255],
    [0,255,255],
    [128,0,0],
    [0,128,0],
    [0,0,128],
    [128,128,0],
    [128,0,128],
    [0,128,128],
    [192,64,0],
    [64,192,0],
    [0,64,192],
    [192,0,64],
    [64,0,192],
    [0,192,64],
    [255,128,0],
    [255,0,128],
    [128,255,0],
    [0,255,128],
    [128,0,255],
    [0,128,255],
    [255,128,128],
    [128,255,128],
    [128,128,255],
    [200,200,200]
], dtype=np.uint8)
from utils import visualize_predictions
num_test_samples = 10
_, axes = plt.subplots(num_test_samples, 3, figsize=(3*6, num_test_samples * 4))
visualize_predictions(model, test_set, axes, device, numTestSamples=num_test_samples, 
                      id_to_color = id_to_color)

"""
rgb_image, label = train_set[np.random.choice(len(train_set))]
rgb_image = inverse_transform(rgb_image).permute(1, 2, 0).cpu().detach().numpy()
label = label.cpu().detach().numpy()

# plot sample image
fig, axes = plt.subplots(1,2, figsize=(20,10))
axes[0].imshow(rgb_image)
axes[0].set_title("Image")
axes[0].axis('off')
axes[1].imshow(label)
axes[1].set_title("Label")
axes[1].axis('off')
plt.savefig("sample_image_label.png")
plt.close()
"""