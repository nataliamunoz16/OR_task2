# basic imports
import numpy as np
# DL library imports
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# dataset loader
targetWidth = 512
targetHeight = 768
transform = T.Compose([T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
train_set = FashionDataset(config.TRAIN_IMG, config.TRAIN_MASK, targetHeight, targetWidth, transform)
val_test_set = FashionDataset(config.TEST_IMG, config.TEST_MASK, targetHeight, targetWidth, transform)
val_size = int(0.5 * len(val_test_set))
test_size = len(val_test_set) - val_size
val_set, test_set = random_split(val_test_set, [val_size, test_size])
train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_set, val_set, test_set)

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
