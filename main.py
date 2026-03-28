import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from dataset import FashionDataset, FashionDatasetMaskRCNN
import config
from torch.utils.data import random_split
import torchvision.transforms as T
from Segformer import segformer_mit_b3
from deeplabv3plus import deeplabv3plus
from unet import unet
from utils import (get_dataloaders, train_validate_model, evaluate_model, meanIoU, visualize_predictions, train_one_epoch_maskrcnn, evaluate_maskrcnn)
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import matplotlib.pyplot as plt
import random
import time
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

FULL_CLASSES = ('shirt, blouse','top, t-shirt, sweatshirt','sweater','cardigan','jacket','vest','pants','shorts','skirt','coat','dress','jumpsuit','cape','glasses','hat','headband, head covering, hair accessory','tie','glove','watch','belt','leg warmer','tights, stockings','sock','shoe','bag, wallet','scarf','umbrella','hood','collar','lapel','epaulette','sleeve','pocket','neckline','buckle','zipper','applique','bead','bow','flower','fringe','ribbon','rivet','ruffle','sequin','tassel')
MAIN_CLASSES = FULL_CLASSES[:27]
NUM_CLASSES = len(MAIN_CLASSES) + 1 #+1 for background

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def build_transforms():
    return T.Compose([T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

def build_datasets(target_height, target_width, transform, model_name):
    if model_name == 'maskRCNN':
        train_set_full = FashionDatasetMaskRCNN(config.TRAIN_IMG,config.ANNOTATIONS_TRAIN,transform, target_height,target_width)
        test_set = FashionDatasetMaskRCNN(config.TEST_IMG,config.ANNOTATIONS_TEST, transform, target_height,target_width)
    else:
        train_set_full = FashionDataset(config.TRAIN_IMG,config.TRAIN_MASK,target_height,target_width,transform)
        test_set = FashionDataset(config.TEST_IMG,config.TEST_MASK,target_height,target_width,transform)

    val_size = len(test_set)
    if val_size > len(train_set_full):
        raise ValueError(f"Validation size ({val_size}) is larger than train set ({len(train_set_full)}).")

    train_size = len(train_set_full) - val_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(train_set_full,[train_size, val_size],generator=generator)
    return train_set, val_set, test_set

def build_model(model_name: str, num_classes: int, device: torch.device, pretrained: bool = True):
    if model_name == "deeplabv3+":
        model = deeplabv3plus(num_classes, pretrained=pretrained)
    elif model_name == "segformer":
        model = segformer_mit_b3(in_channels=3, num_classes=num_classes)
        if pretrained:
            weights_path = "segformers/segformer_mit_b3_imagenet_weights.pt"
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=device)
            else:
                print(f"Warning: pretrained weights not found at {weights_path}")
    elif model_name == "maskRCNN":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        model.backbone.load_state_dict(state_dict)
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, NUM_CLASSES)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)
    elif model_name == "unet":
        model =unet(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model.to(device)

def get_id_to_color():
    return np.array([
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
        [128,128,255]
    ], dtype=np.uint8)

def main():
    set_seed(42)

    target_width = 384
    target_height = 384
    n_epochs = 20
    base_lr = 0.5
    batch_size = 20
    pretrained = True
    data_augmentation = False
    model_name = "unet"
    model_file = f"{model_name}_{target_height}_{target_width}_{base_lr}_{batch_size}"
    suffixes = []
    if pretrained:
        suffixes.append("pretrained")
    if data_augmentation:
        suffixes.append("data_aug")
    if suffixes:
        model_file += "_" + "_".join(suffixes)

    device = get_device()
    print(f"Using device: {device}")

    transform = build_transforms()
    train_set, val_set, test_set = build_datasets(target_height,target_width,transform, model_name)
    train_loader, val_loader, test_loader = get_dataloaders(train_set, val_set, test_set, batch_size=batch_size)
    model = build_model(model_name, NUM_CLASSES, device, pretrained=pretrained)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-4)
    #scheduler = OneCycleLR(optimizer, max_lr= max_lr, epochs = n_epochs,steps_per_epoch = len(train_loader), pct_start=0.3, div_factor=10, anneal_strategy='cos')
    warmup_epochs = 2

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs * len(train_loader)
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(n_epochs - warmup_epochs) * len(train_loader),
        eta_min=1e-6
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs * len(train_loader)]
    )
    results = {}
    start_train_val = time.time()
    _ = train_validate_model(model, n_epochs, model_name, criterion, optimizer, 
                         device, train_loader, val_loader,NUM_CLASSES, lr_scheduler = scheduler, output_path = config.MODELS, model_name_save=model_file)
    results['train_time'] = time.time() - start_train_val
    model_path_best_loss = os.path.join(config.MODELS, f"{model_file}_best_loss.pt")
    if not os.path.exists(model_path_best_loss):
        raise FileNotFoundError(f"Checkpoint not found for best loss: {model_path_best_loss}")
    model.load_state_dict(torch.load(model_path_best_loss, map_location=device))
    test_metrics = evaluate_model(model, test_loader, criterion, NUM_CLASSES, device)
    print(f"\nModel has {test_metrics['mDice']} mean Dice in test set")
    result_best_loss = {
            "mIoU": float(test_metrics["mIoU"]),
            "mDice": float(test_metrics["mDice"]),
            "mDice_no_bg": float(test_metrics["mDice_no_bg"]),
            "accuracy": float(test_metrics["accuracy"]),
            "mean_acc": float(test_metrics["mean_acc"]),
            "mean_acc_no_bg": float(test_metrics["mean_acc_no_bg"]),
            "dice_per_class": test_metrics["dice_per_class"].tolist(),
            "accuracy_per_class": test_metrics["accuracy_per_class"].tolist(),
            "iou_per_class": test_metrics["iou_per_class"].tolist(),
        }
    results['best loss model'] = result_best_loss


    model_path_mdice = os.path.join(config.MODELS, f"{model_file}_best_mDice.pt")
    if not os.path.exists(model_path_mdice):
        raise FileNotFoundError(f"Checkpoint not found for mDice: {model_path_mdice}")
    model.load_state_dict(torch.load(model_path_mdice, map_location=device))
    test_metrics = evaluate_model(model, test_loader, criterion, NUM_CLASSES, device)
    print(f"\nModel has {test_metrics['mDice']} best mean Dice in test set")
    result_mdice = {
            "mIoU": float(test_metrics["mIoU"]),
            "mDice": float(test_metrics["mDice"]),
            "mDice_no_bg": float(test_metrics["mDice_no_bg"]),
            "accuracy": float(test_metrics["accuracy"]),
            "mean_acc": float(test_metrics["mean_acc"]),
            "mean_acc_no_bg": float(test_metrics["mean_acc_no_bg"]),
            "dice_per_class": test_metrics["dice_per_class"].tolist(),
            "accuracy_per_class": test_metrics["accuracy_per_class"].tolist(),
            "iou_per_class": test_metrics["iou_per_class"].tolist(),
        }
    results['mDice model'] = result_mdice
    metrics_json_path = os.path.join(config.RESULTS, f"{model_file}_test_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(results, f, indent=4)


    #id_to_color = np.array([[i,i,i] for i in range(28)], dtype=np.uint8)
    id_to_color = get_id_to_color()
    num_test_samples = 10
    _, axes = plt.subplots(num_test_samples, 3, figsize=(3*6, num_test_samples * 4))
    visualize_predictions(model, test_set, axes, device, numTestSamples=num_test_samples, 
                        id_to_color = id_to_color, model_name_save = model_file)

if __name__ == '__main__':
    main()