
import config
import os
import matplotlib.pyplot as plt
import cv2
import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import random_split
from pycocotools.coco import COCO
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from dataset import FashionDatasetMaskRCNN
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils_maskrcnn

if torch.cuda.is_available():
    device=torch.device("cuda")
else:
    device=torch.device("cpu")

print(device)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils_maskrcnn.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils_maskrcnn.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        #with torch.cuda.amp.autocast(enabled=scaler is not None):
        with torch.amp.autocast(enabled=scaler is not None, device_type=str(device)):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils_maskrcnn.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def evaluate_model(model, dataloader, num_classes, device):
    model.eval()
    total_loss=0.0

    #confusion matrix
    print(num_classes)
    conf_mat=torch.zeros((num_classes, num_classes), dtype=torch.int64, device = device)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, total=len(dataloader)):

            #GET OUTPUT MASKS
            images= images[0].to(device)
            imc, imh, imw = images.shape
            y_preds = model([images])#a
            y_preds = y_preds[0]
            if len(y_preds["masks"])==0: 
                print("skipping, empty prediction: model returned nothing")
                continue
            y_preds_masks = y_preds["masks"][0]
            
            
            #REFORMAT THE OUTPUT
            Np = y_preds_masks.shape[0] #numcategories_in_prediction
            y_preds_masks.reshape((Np,imh*imw)) 
            threshold = 0.5
            over = y_preds_masks > threshold #True if >0.5, else false
            over = over.long() #convert to 0s and 1s
            preds = y_preds_masks*over #set all values under 0.5 to zero
            
            #get max across categories, and corresponding category
            preds, cats = torch.max(preds, axis=0) 
            # preds: [0,0.7,0.3,0,0,0,0.3,0.2,... etc]
            # cats: [0,1,0,0,2,0,1,0,4,0,2] (fins a N sent N les 
            # CATEGORIES DIFERENTS A LA FOTO, no les totals)
            preds = [1 if pred!=0 else 0 for pred in preds]
            # [0,1,1,0,0,0,1,1,...etc]
            y_preds_labels = y_preds["labels"]
            # [0,1,23,23,4] 
            whichcats = [y_preds_labels[cat] for cat in cats]
            # [0,1,0,0,2,0,1,0,4...] -> [0,23,0,0,8,0,23,0,7,...]
            preds*=whichcats #fa falta? poder no! és el mateix al final
            preds = preds.long() #idem
            
            #REFORMAT THE TARGET
            labels = labels[0]
            N = labels["masks"].shape[0] #numcategories_in_target
            A = labels["masks"].reshape((N,imh*imw))
            e = torch.tensor(np.zeros([imh*imw]))
            for i in range(len(A)):
                k = int(labels["labels"][i])
                Aik = A[i]*k
                e = e+Aik #assumim que target no té overlapping masks           
            preds= preds.view(-1)
            labels= e.view(-1)

            for t, p in zip(labels, preds):
                if t >= 28 or p>= 28:
                    print(t,p)
                    continue
                conf_mat[t.long(), p.long()] += 1
    
    conf_mat= conf_mat.float()
    #print("confmat",conf_mat.shape,conf_mat)
    TP= torch.diag(conf_mat)
    FP= conf_mat.sum(dim=0)-TP
    FN= conf_mat.sum(dim=1)-TP
    TN= conf_mat.sum()-(TP+FP+FN)

    # Dice per class
    dice_per_class = (2*TP)/(2*TP+FP+FN+1e-6)
    mDice = dice_per_class.mean().item()
    mDice_no_bg = dice_per_class[1:].mean().item()

    # Accuracy per class
    acc_per_class= TP/(TP + FN + 1e-6)
    mAcc= acc_per_class.mean().item()
    mAcc_no_bg= acc_per_class[1:].mean().item()

    # Global accuracy
    accuracy= TP.sum()/conf_mat.sum()

    # IoU
    iou_per_class=TP/(TP+FP+FN+1e-6)
    mIoU= iou_per_class.mean().item()
    #evaluation_loss= total_loss/len(dataloader)
    #return {"loss": evaluation_loss,"mIoU": mIoU, "mDice": mDice,"mDice_no_bg": mDice_no_bg,"accuracy": accuracy.item(),"dice_per_class": dice_per_class.cpu().numpy(),"mean_acc": mAcc,"mean_acc_no_bg": mAcc_no_bg,"accuracy_per_class": acc_per_class.cpu().numpy(),"iou_per_class": iou_per_class.cpu().numpy()}
    return {"mIoU": mIoU, "mDice": mDice,"mDice_no_bg": mDice_no_bg,"accuracy": accuracy.item(),"dice_per_class": dice_per_class.cpu().numpy(),"mean_acc": mAcc,"mean_acc_no_bg": mAcc_no_bg,"accuracy_per_class": acc_per_class.cpu().numpy(),"iou_per_class": iou_per_class.cpu().numpy(), "conf_mat":conf_mat}



def get_transform():
    return ToTensor()

if __name__ == "__main__":
    train_dir = config.TRAIN_IMG
    train_annotation = config.ANNOTATIONS_TRAIN
    seg_train = config.TRAIN_MASK
    val_dir = config.TEST_MASK
    val_annotation = config.ANNOTATIONS_TEST
    seg_val = config.TEST_MASK

    TARGETH = 384
    TARGETW = 384


    print('Train contains:', len(os.listdir(train_dir)), 'images')
    print('Seg_train contains:',len(os.listdir(seg_train)), 'images')
    print('Validation contains:',len(os.listdir(val_dir)), 'images')
    print('Seg_val contains:',len(os.listdir(seg_val)), 'images')

    train_dataset = FashionDatasetMaskRCNN(
        root_dir=train_dir,
        annotation_file=train_annotation,
        transforms=get_transform(),
        target_height=TARGETH,
        target_width=TARGETW,
        )

    val_dataset = FashionDatasetMaskRCNN(
        root_dir=val_dir,
        annotation_file=val_annotation,
        transforms=get_transform(),
        target_height=TARGETH,
        target_width=TARGETW,
    )

    r = 1 #train dataset fraction ratio
    v = 1 #validation dataset ratio
    train_dataset_reduced,_ = random_split(train_dataset,[r,1-r])
    val_dataset_reduced,_ = random_split(val_dataset,[v,1-v])
    train_loader = DataLoader(train_dataset_reduced, batch_size=10, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset_reduced, batch_size=10, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    FULL_CLASSES = [' ','shirt, blouse','top, t-shirt, sweatshirt','sweater','cardigan',
                'jacket','vest','pants','shorts','skirt','coat','dress','jumpsuit',
                'cape','glasses','hat','headband, head covering, hair accessory',
                'tie','glove','watch','belt','leg warmer','tights, stockings','sock',
                'shoe','bag, wallet','scarf','umbrella','hood','collar','lapel',
                'epaulette','sleeve','pocket','neckline','buckle','zipper','applique',
                'bead','bow','flower','fringe','ribbon','rivet','ruffle','sequin','tassel']
    CLASS_NAMES = FULL_CLASSES[:28]

    images, targets = next(iter(train_loader))
    for i in range(len(images)):
        # CxHxW --> HxWxC
        image = images[i].permute(1, 2, 0).cpu().numpy()
        # Rescale
        image = (image * 255).astype(np.uint8)
        # Convert RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        overlay = image.copy()
        
        # Extract masks, bounding boxes, and labels for the current image
        masks = targets[i]['masks'].cpu().numpy()
        boxes = targets[i]['boxes'].cpu().numpy()
        labels = targets[i]['labels'].cpu().numpy()
    
        for j in range(len(masks)):
            mask = masks[j]
            box = boxes[j]
            label_id = labels[j]
    
            # Get class name from mapping
            class_name = CLASS_NAMES[label_id]  # assuming 1-based labels #+1
    
            # Random color
            color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
    
            # Alpha blend mask
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = mask * color[c]
            alpha = 0.4
            overlay = np.where(mask[..., None], 
                            ((1 - alpha) * overlay + alpha * colored_mask).astype(np.uint8), 
                            overlay)
    
            # Draw label
            x1, y1, x2, y2 = map(int, box)
            cv2.putText(overlay, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 1, lineType=cv2.LINE_AA)
    
    
        # Show the result
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Sample {i + 1}")
        #plt.show()
    
    # Load a pre-trained Mask R-CNN model with a ResNet-50 FPN backbone
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    # Determine the number of classes (including background) from training dataset
    # COCO format includes category IDs, and we add +1 for background
    num_classes = 28 #len(train_dataset.coco.getCatIds()) + 1
    
    # Replace the existing box predictor with a new one for our number of classes
    # in_features_box: number of input features to the classification layer
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    
    # Replace the existing mask predictor with a new one for our number of classes
    # in_features_mask: number of input channels to the first convolutional layer 
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    # Move the model to the specified device (e.g., GPU) for training or inference
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
 
    # Define the optimizer SGD(Stochastic Gradient Descent) 
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    
    num_epochs = 20
 
    # Loop through each epoch
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
    
        # Train the model for one epoch, printing status every 25 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=25)  # Using train_loader for training
    

        results = evaluate_model(model, val_loader, num_classes, device)
        print(results)
        # Evaluate the model on the validation dataset
        #evaluate(model, val_loader, num_classes, device=device)  # Using val_loader for evaluation
    
        # Optionally, save the model checkpoint after each epoch
        torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
