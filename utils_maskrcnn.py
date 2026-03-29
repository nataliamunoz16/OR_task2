# basic imports
import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import namedtuple
import config

# DL library imports
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

# For dice loss function
import segmentation_models_pytorch as smp

# for interactive widgets
#import IPython.display as Disp
# #from ipywidgets import widgets

###################################
# FILE CONSTANTS
###################################

# Convert to torch tensor and normalize images using Imagenet values
preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])

# when using torch datasets we defined earlier, the output image
# is normalized. So we're defining an inverse transformation to 
# transform to normal RGB format
inverse_transform = transforms.Compose([
        transforms.Normalize(mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225), std=(1/0.229, 1/0.224, 1/0.225))
    ])


# Constants for Standard color mapping
# reference : https://github.com/bdd100k/bdd100k/blob/master/bdd100k/label/label.py

Label = namedtuple( "Label", [ "name", "train_id", "color"])
drivables = [ 
             Label("direct", 0, (219, 94, 86)),        # red
             Label("alternative", 1, (86, 211, 219)),  # cyan
             Label("background", 2, (0, 0, 0)),        # black          
            ]
train_id_to_color = [c.color for c in drivables if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color = np.array(train_id_to_color)



#####################################
### ROI SELECT CLASS DEFINITION ##
#####################################

# class roi_select():
#     def __init__(self,im, figsize=(12,6), line_color=(255,0,0)):
#         # class variables
#         self.im = im
#         self.selected_points = []
#         self.line_color = line_color

#         # plot input image, store figure handles
#         self.fig,ax = plt.subplots(figsize=figsize)
#         self.img = ax.imshow(self.im.copy())
#         plt.suptitle('Select Rectangular ROI')

#         # connect event handlers
#         self.ka = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
#         disconnect_button = widgets.Button(description="Disconnect mpl")
#         Disp.display(disconnect_button)
#         disconnect_button.on_click(self.disconnect_mpl)
        
#     def draw_roi_on_img(self,img,pts):
#         # plot polygon edges in single color 
#         pts = np.array(pts, np.int32)
#         pts = pts.reshape((-1,2))
#         # cv2.polylines(img, [pts], True, self.line_color, 5)
#         cv2.rectangle(img, pts[0], pts[1], self.line_color, 5)
#         return img

#     def onclick(self, event):
#         self.selected_points.append([event.xdata,event.ydata])
#         if len(self.selected_points) == 2:
#             self.img.set_data(self.draw_roi_on_img(self.im.copy(),self.selected_points))
#             self.disconnect_mpl(None)

#     def disconnect_mpl(self,_):
#         self.fig.canvas.mpl_disconnect(self.ka)


#     def get_bbox_indices(self, scale_factor = None):
#         pts = np.array(self.selected_points)
#         if(scale_factor is not None):
#             pts = pts * scale_factor

#         # bounding box coordinates as indices 
#         # min_x, min_y is top left index
#         # max_x, max_y is bottom right index
#         roi_indices = pts.astype(int)
#         indices = {}
#         indices['min_x'], indices['min_y'] = np.min(roi_indices, axis=0)
#         indices['max_x'], indices['max_y'] = np.max(roi_indices, axis=0)
#         return indices


#####################################
### TORCH DATASET CLASS DEFINITION ##
#####################################


class BDD100k_dataset(Dataset):
    def __init__(self, images, labels, tf=None):
        """Dataset class for BDD100k_dataset drivable / segmentation data """
        self.images = images
        self.labels = labels
        self.tf = tf
    
    def __len__(self):
        return self.images.shape[0]
  
    def __getitem__(self, index):
        # read source image and convert to RGB, apply transform
        rgb_image = self.images[index]
        if self.tf is not None:
            rgb_image = self.tf(rgb_image)

        # read label image and convert to torch tensor
        label_image  = torch.from_numpy(self.labels[index]).long()
        return rgb_image, label_image  



###################################
# FUNCTION TO GET TORCH DATASET  #
###################################

def get_datasets(images, labels):
    data = BDD100k_dataset(images, labels, tf=preprocess)

    # split train data into train, validation and test sets
    total_count = len(data)
    train_count = int(0.7 * total_count) 
    valid_count = int(0.2 * total_count)
    test_count = total_count - train_count - valid_count
    train_set, val_set, test_set = torch.utils.data.random_split(data, 
                (train_count, valid_count, test_count), generator=torch.Generator().manual_seed(1))
    return train_set, val_set, test_set


###################################
# FUNCTION TO GET TORCH DATALOADER  #
###################################

def get_dataloaders(train_set, val_set, test_set, batch_size=8):
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,drop_last=True)
    val_dataloader   = DataLoader(val_set, batch_size=batch_size)
    test_dataloader  = DataLoader(test_set, batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader    



###################################
# METRIC CLASS DEFINITION
###################################

class meanIoU:
    """ Class to find the mean IoU using confusion matrix approach """    
    def __init__(self, num_classes):
        self.iou_metric = 0.0
        self.num_classes = num_classes
        # placeholder for confusion matrix on entire dataset
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, y_preds, labels):
        """ Function finds the IoU for the input batch
        and add batch metrics to overall metrics """
        predicted_labels = torch.argmax(y_preds, dim=1)
        batch_confusion_matrix = self._fast_hist(labels.numpy().flatten(), predicted_labels.numpy().flatten())
        self.confusion_matrix += batch_confusion_matrix
    
    def _fast_hist(self, label_true, label_pred):
        """ Function to calculate confusion matrix on single batch """
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def compute(self):
        """ Computes overall meanIoU metric from confusion matrix data """ 
        hist = self.confusion_matrix
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        return mean_iu

    def reset(self):
        self.iou_metric = 0.0
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))



###################################
# PSPNET LOSS FUNCTION DEFINITION
###################################

class pspnet_loss(nn.Module):
    def __init__(self, num_classes, aux_weight):
        super(pspnet_loss, self).__init__()
        self.aux_weight = aux_weight
        self.loss_fn = smp.losses.DiceLoss('multiclass', 
                        classes=np.arange(num_classes).tolist(), log_loss = True, smooth=1.0)
    
    def forward(self, preds, labels):
        # if input predictions is in dict format
        # calculate total loss as weighted sum of 
        # main and auxiliary losses
        if(isinstance(preds, dict) == True):
            main_loss = self.loss_fn(preds['main'], labels)
            aux_loss = self.loss_fn(preds['aux'], labels)
            loss = (1 - self.aux_weight) * main_loss + self.aux_weight * aux_loss
        else:
            loss = self.loss_fn(preds, labels)
        return loss



###################################
# POLY LR DECAY SCHEDULER DEFINITION
###################################

class polynomial_lr_decay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    
    Reference:
        https://github.com/cmpark0126/pytorch-polynomial-lr-decay
    """
    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) * 
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                self.end_learning_rate for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) * 
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr


###################################
# FUNCTION TO PLOT TRAINING, VALIDATION CURVES
###################################


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



###################################
# FUNCTION TO EVALUATE MODEL ON DATALOADER
###################################
"""
def evaluate_model(model, dataloader, criterion, metric_class, num_classes, device):
    model.eval()
    total_loss = 0.0
    metric_object = metric_class(num_classes)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)                
            y_preds = model(inputs)

            # calculate loss
            loss = criterion(y_preds, labels)
            total_loss += loss.item()

            # update batch metric information            
            metric_object.update(y_preds.cpu().detach(), labels.cpu().detach())

    evaluation_loss = total_loss / len(dataloader)
    evaluation_metric = metric_object.compute()
    return evaluation_loss, evaluation_metric
"""

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



###################################
# FUNCTION TO TRAIN, VALIDATE MODEL ON DATALOADER
###################################

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


###################################
# FUNCTION TO VISUALIZE MODEL PREDICTIONS
###################################

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

        # predicted label image
        y_pred = torch.argmax(model(inputImage.unsqueeze(0)), dim=1).squeeze(0)
        label_class_predicted = y_pred.cpu().detach().numpy()    
        axes[i, 2].imshow(id_to_color[label_class_predicted])
        axes[i, 2].set_title("Predicted Label")

    #plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS, f"{model_name_save}_predicted.png"))
    plt.close()



###################################
# FUNCTION TO VISUALIZE MODEL 
# PREDICTIONS ON TEST VIDEO
###################################

def predict_video(model, model_name, input_video_path, output_dir, 
            target_width, target_height, device):
    file_name = input_video_path.split(os.sep)[-1].split('.')[0]
    output_filename = f'{file_name}_{model_name}_output.avi'
    output_video_path = os.path.join(output_dir, *[output_filename])

    # handles for input output videos
    input_handle = cv2.VideoCapture(input_video_path)
    output_handle = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), \
                                    30, (target_width, target_height))

    # create progress bar
    num_frames = int(input_handle.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total = num_frames, position=0, leave=True)

    while(input_handle.isOpened()):
        ret, frame = input_handle.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # create torch tensor to give as input to model
            pt_image = preprocess(frame)
            pt_image = pt_image.to(device)

            # get model prediction and convert to corresponding color
            y_pred = torch.argmax(model(pt_image.unsqueeze(0)), dim=1).squeeze(0)
            predicted_labels = y_pred.cpu().detach().numpy()
            cm_labels = (train_id_to_color[predicted_labels]).astype(np.uint8)

            # overlay prediction over input frame
            overlay_image = cv2.addWeighted(frame, 1, cm_labels, 0.25, 0)
            overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)

            # write output result and update progress
            output_handle.write(overlay_image)
            pbar.update(1)

        else:
            break

    output_handle.release()
    input_handle.release()



import datetime
import errno
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def collate_fn(batch):
    return tuple(zip(*batch))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)