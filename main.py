import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from einops import rearrange
from timm.models.layers import drop_path, trunc_normal_
from utils import get_dataloaders, inverse_transform
from dataset import FashionDataset
import config
from torch.utils.data import random_split
import torchvision.transforms as T

from deeplabv3plus import deeplabv3plus
from utils import *


def main():
    FULL_CLASSES = ('shirt, blouse','top, t-shirt, sweatshirt','sweater','cardigan','jacket','vest','pants','shorts','skirt','coat','dress','jumpsuit','cape','glasses','hat','headband, head covering, hair accessory','tie','glove','watch','belt','leg warmer','tights, stockings','sock','shoe','bag, wallet','scarf','umbrella','hood','collar','lapel','epaulette','sleeve','pocket','neckline','buckle','zipper','applique','bead','bow','flower','fringe','ribbon','rivet','ruffle','sequin','tassel')
    MAIN_CLASSES = FULL_CLASSES[:27]
    num_classes = len(MAIN_CLASSES) + 1

    # dataset loader
    targetWidth = 512
    targetHeight = 768

    N_EPOCHS = 30
    NUM_CLASSES = 28
    MAX_LR = 1e-3
    model_name = ['maskRCNN', 'deeplabv3+', 'segformer'][1]
    MODEL_NAME = f'{model_name}_{targetHeight}_{targetWidth}_CE_loss'

    ignore_index = 255

    transform = T.Compose([T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    #train_set = FashionDataset(config.TRAIN_IMG, config.TRAIN_MASK, targetHeight, targetWidth, transform) # no hem pogut descarregar el file
    train_set = FashionDataset(config.TRAIN_IMG, config.TRAIN_MASK, targetHeight, targetWidth, transform) 
    val_test_set = FashionDataset(config.TEST_IMG, config.TEST_MASK, targetHeight, targetWidth, transform)
    val_size = int(0.5 * len(val_test_set))
    test_size = len(val_test_set) - val_size
    val_set, test_set = random_split(val_test_set, [val_size, test_size])
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_set, val_set, test_set)


    if model_name == 'maskRCNN':
        pass
    elif model_name == 'deeplabv3+':
        model = deeplabv3plus(num_classes)
    elif model_name == 'segformer':
        pass
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")


    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
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

if __name__ == '__main__':
    main()