import torch
import segmentation_models_pytorch as smp

def deeplabv3plus(num_classes):
    return smp.DeepLabV3Plus(encoder_name="resnet50",encoder_weights="imagenet",in_channels=3,classes=num_classes,)