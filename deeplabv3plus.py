import segmentation_models_pytorch as smp

def deeplabv3plus(num_classes, pretrained):
    weights = "imagenet" if pretrained else None
    return smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=weights, in_channels=3, classes=num_classes,)
