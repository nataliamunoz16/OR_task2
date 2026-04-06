import os
import numpy
import config
import numpy as np
import torch
import json
import torchvision.transforms as T
from dataset import FashionDataset
from torch.utils.data import random_split

FULL_CLASSES = ('background','shirt, blouse','top, t-shirt, sweatshirt','sweater','cardigan','jacket','vest','pants','shorts','skirt','coat','dress','jumpsuit','cape','glasses','hat','headband, head covering, hair accessory','tie','glove','watch','belt','leg warmer','tights, stockings','sock','shoe','bag, wallet','scarf','umbrella','hood','collar','lapel','epaulette','sleeve','pocket','neckline','buckle','zipper','applique','bead','bow','flower','fringe','ribbon','rivet','ruffle','sequin','tassel')


def build_transforms():
    return T.Compose([T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

def compute_label_distribution_train(dataset, num_classes=28):
    pixel_counts={i: [0, 0] for i in range(num_classes)}
    for i in range(len(dataset)):
        _, mask=dataset[i]
        mask_np = mask.numpy()
        vals,counts= np.unique(mask_np, return_counts=True)
        for value, count in zip(vals,counts):
            pixel_counts[value][0] = pixel_counts[value][0] + count
            pixel_counts[value][1] = pixel_counts[value][1] + 1
    print("Label distribution statistics:")
    for class_id in pixel_counts.keys():
        print(f"Label: {FULL_CLASSES[class_id]};  Number of instances: {pixel_counts[class_id][1]};  pixels per instance: {pixel_counts[class_id][0]/pixel_counts[class_id][1]:.2f}")
    
    return pixel_counts

def overrepresented(best_model_path, background=True, number_of_instances=1000, min_dice=0.7):
    #Pixel label distribution
    transform = build_transforms()
    train_set_full = FashionDataset(config.TRAIN_IMG,config.TRAIN_MASK,384,384,transform)
    test_set = FashionDataset(config.TEST_IMG,config.TEST_MASK,384,384,transform)

    val_size = len(test_set)
    if val_size > len(train_set_full):
        raise ValueError(f"Validation size ({val_size}) is larger than train set ({len(train_set_full)}).")

    train_size = len(train_set_full) - val_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(train_set_full,[train_size, val_size],generator=generator)
    data = compute_label_distribution_train(train_set)

    #Metrics
    with open(best_model_path, "w") as f:
        metrics = json.load(f)
    for i, k in enumerate(data.keys()):
        data[k].append(metrics[18]["dice_per_class"][i])
        data[k].append(metrics[18]["accuracy_per_class"][i])
        data[k].append(metrics[18]["iou_per_class"][i])
    
    #select_overrepresented
    overrepresented_ids = []
    if not background:
        overrepresented_ids.append(0)
    for i in range(1, len(data.keys())):
        if data[data.keys()[i]][1]>number_of_instances and data[data.keys()[i]][2]>min_dice:
            overrepresented_ids.append(data.keys()[i])

if __name__ == "__main__":
    transform = build_transforms()
    train_set_full = FashionDataset(config.TRAIN_IMG,config.TRAIN_MASK,384,384,transform)
    test_set = FashionDataset(config.TEST_IMG,config.TEST_MASK,384,384,transform)

    val_size = len(test_set)
    if val_size > len(train_set_full):
        raise ValueError(f"Validation size ({val_size}) is larger than train set ({len(train_set_full)}).")

    train_size = len(train_set_full) - val_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(train_set_full,[train_size, val_size],generator=generator)
    compute_label_distribution_train(train_set)
