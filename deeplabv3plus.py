import torch
import segmentation_models_pytorch as smp

def deeplabv3plus(num_classes, pretrained=False, in_channels=3):
    if not pretrained:
        return smp.DeepLabV3Plus(encoder_name="resnet50",in_channels=in_channels,classes=num_classes,)
    return smp.DeepLabV3Plus(encoder_name="resnet50",encoder_weights="imagenet",in_channels=in_channels,classes=num_classes,)
