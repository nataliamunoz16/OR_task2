# Utility functions for data loading, training, evaluation, metrics, plotting and visualization.
# basic imports
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import namedtuple
import config
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

"""
Based on: https://colab.research.google.com/gist/Jeremy26/13f71c273f0a1a93f758d02b2b77802e/segformer-cityscapes-run.ipynb?authuser=0#scrollTo=ae77a0b4
"""

def get_dataloaders(train_set, val_set, test_set, batch_size=8):
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,drop_last=True)
    val_dataloader   = DataLoader(val_set, batch_size=batch_size)
    test_dataloader  = DataLoader(test_set, batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader  

def evaluate_model(model, dataloader, criterion, num_classes, device):
    model.eval()
    total_loss=0.0

    #confusion matrix
    conf_mat=torch.zeros((num_classes, num_classes), dtype=torch.int64, device = device)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, total=len(dataloader)):
            inputs=inputs.to(device)
            labels= labels.to(device)
            y_preds= model(inputs)

            #loss
            loss= criterion(y_preds, labels)
            total_loss += loss.item()

            #predictions
            preds= torch.argmax(y_preds, dim=1)

            # flatten
            preds= preds.view(-1)
            labels= labels.view(-1)
            
            idx = labels * num_classes + preds
            conf_mat += torch.bincount(idx,minlength=num_classes * num_classes).reshape(num_classes, num_classes)
            #update confusion matrix
            #for t, p in zip(labels, preds):
                #conf_mat[t.long(), p.long()] += 1

    conf_mat= conf_mat.float()
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
    evaluation_loss= total_loss/len(dataloader)
    return {"loss": evaluation_loss,"mIoU": mIoU, "mDice": mDice,"mDice_no_bg": mDice_no_bg,"accuracy": accuracy.item(),"dice_per_class": dice_per_class.cpu().numpy(),"mean_acc": mAcc,"mean_acc_no_bg": mAcc_no_bg,"accuracy_per_class": acc_per_class.cpu().numpy(),"iou_per_class": iou_per_class.cpu().numpy()}

def plot_training_results(df, model_name_save):
    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.set_ylabel('trainLoss', color='tab:red')
    ax1.plot(df['epoch'].values, df['trainLoss'].values, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()  
    ax2.set_ylabel('validationLoss', color='tab:blue')
    ax2.plot(df['epoch'].values, df['validationLoss'].values, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.suptitle(f'{model_name_save} Training, Validation Curves')
    plt.savefig(os.path.join(config.RESULTS, f"{model_name_save}_losses.png"))

def train_validate_model(model, num_epochs, model_name, criterion, optimizer, 
                         device, dataloader_train, dataloader_valid, num_classes, lr_scheduler = None,
                         output_path = '.', model_name_save=None,early_stopping_patience = 10, min_delta=1e-5):
    # initialize placeholders for running values
    results = []    
    min_val_loss = np.inf
    max_val_mDice = -np.inf
    len_train_loader = len(dataloader_train)

    os.makedirs(output_path, exist_ok=True)

    epochs_no_improve = 0

    # move model to device
    model.to(device)

    if model_name_save is None:
        model_name_save = model_name
    
    for epoch in range(num_epochs):
        print(f"Starting {epoch + 1} epoch ...")
        
        # Training
        model.train()
        train_loss = 0.0
        len_train_loader = len(dataloader_train)
        for inputs, labels in tqdm(dataloader_train, total=len_train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device) 
            valid_pixels=(labels!= 255).sum().item()
            if valid_pixels== 0:
                print("Batch with 0 valid pixels")
                continue
            # Forward pass
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
            train_loss += loss.item()
              
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # adjust learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()
            
        # compute per batch losses, metric value
        train_loss = train_loss / len_train_loader
        val_metrics = evaluate_model(
                        model, dataloader_valid, criterion, num_classes, device)
        
        validation_loss = val_metrics["loss"]
        validation_mIoU = val_metrics["mIoU"]
        validation_mDice = val_metrics["mDice"]
        validation_mAcc = val_metrics["mean_acc"]
        print(f'Epoch: {epoch+1}, trainLoss:{train_loss:6.5f}, validationLoss:{validation_loss:6.5f}, validation_mIoU:{validation_mIoU: 4.2f}, validation_mDice:{validation_mDice: 4.2f}, validation_mAcc:{validation_mAcc: 4.2f}')


        epoch_result = {
            "epoch": epoch + 1,
            "trainLoss": float(train_loss),
            "validationLoss": float(validation_loss),
            "mIoU": float(validation_mIoU),
            "mDice": float(validation_mDice),
            "mDice_no_bg": float(val_metrics["mDice_no_bg"]),
            "accuracy": float(val_metrics["accuracy"]),
            "mean_acc": float(validation_mAcc),
            "mean_acc_no_bg": float(val_metrics["mean_acc_no_bg"]),
            "dice_per_class": val_metrics["dice_per_class"].tolist(),
            "accuracy_per_class": val_metrics["accuracy_per_class"].tolist(),
            "iou_per_class": val_metrics["iou_per_class"].tolist(),
        }
        results.append(epoch_result)


        # if validation loss has decreased, save model and reset variable
        if validation_loss <= min_val_loss-min_delta:
            min_val_loss = validation_loss
            epochs_no_improve=0
            torch.save(model.state_dict(), os.path.join(output_path, f"{model_name_save}_best_loss.pt"))
            # torch.jit.save(torch.jit.script(model), f"{output_path}/{model_name}.pt")
        else:
            epochs_no_improve +=1
        if validation_mDice >= max_val_mDice:
            max_val_mDice = validation_mDice
            torch.save(model.state_dict(),os.path.join(output_path, f"{model_name_save}_best_mDice.pt"))
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}. "
                f"Validation loss did not improve by more than {min_delta} "
                f"for {early_stopping_patience} consecutive epochs."
            )
            break

    # plot results
    metrics_json_path = os.path.join(config.RESULTS, f"{model_name_save}_validation_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(results, f, indent=4)
    results = pd.DataFrame(results)
    plot_training_results(results, model_name_save)
    return results
Label = namedtuple( "Label", [ "name", "train_id", "color"])
drivables = [ 
             Label("direct", 0, (219, 94, 86)),
             Label("alternative", 1, (86, 211, 219)),
             Label("background", 2, (0, 0, 0)),      
            ]
train_id_to_color = [c.color for c in drivables if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color = np.array(train_id_to_color)
inverse_transform = transforms.Compose([
        transforms.Normalize(mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225), std=(1/0.229, 1/0.224, 1/0.225))
    ])
def visualize_predictions(model : torch.nn.Module, dataSet : Dataset,  
        axes, device :torch.device, numTestSamples : int,
        id_to_color : np.ndarray = train_id_to_color, model_name_save : str = None):
    """Function visualizes predictions of input model on samples from
    cityscapes dataset provided

    Args:
        model (torch.nn.Module): model whose output we're to visualize
        dataSet (Dataset): dataset to take samples from
        device (torch.device): compute device as in GPU, CPU etc
        numTestSamples (int): number of samples to plot
        id_to_color (np.ndarray) : array to map class to colormap
    """
    if model_name_save is None:
        model_name_save = "model"
    model.to(device=device)
    model.eval()

    # predictions on random samples
    testSamples = np.random.choice(len(dataSet), numTestSamples).tolist()
    # _, axes = plt.subplots(numTestSamples, 3, figsize=(3*6, numTestSamples * 4))
    
    for i, sampleID in enumerate(testSamples):
        inputImage, gt = dataSet[sampleID]

        # input rgb image   
        inputImage = inputImage.to(device)
        landscape = inverse_transform(inputImage).permute(1, 2, 0).cpu().detach().numpy()
        axes[i, 0].imshow(landscape)
        axes[i, 0].set_title("Landscape")

        # groundtruth label image
        label_class = gt.cpu().detach().numpy()
        axes[i, 1].imshow(id_to_color[label_class])
        axes[i, 1].set_title("Groudtruth Label")
        if model_name_save.startswith("maskRCNN") or "maskRCNN" in model_name_save:
            preds=model([inputImage.unsqueeze(0).to(device)])[0]
            masks=preds["masks"] > 0.5
            labels=preds["labels"]
            combined=np.zeros((inputImage.shape[1], inputImage.shape[2]), dtype=np.uint8)
            for j in range(len(masks)):
               mask_j=masks[j,0].cpu().numpy()
               combined[mask_j]= labels[j].cpu().item()
            axes[i, 2].imshow(id_to_color[combined])
            axes[i, 2].set_title("Predicted Label")
        else:
            # predicted label image
            y_pred = torch.argmax(model(inputImage.unsqueeze(0)), dim=1).squeeze(0)
            label_class_predicted = y_pred.cpu().detach().numpy()    
            axes[i, 2].imshow(id_to_color[label_class_predicted])
            axes[i, 2].set_title("Predicted Label")

    #plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS, f"{model_name_save}_predicted.png"))
    plt.close()


