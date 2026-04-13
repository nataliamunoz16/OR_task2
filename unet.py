# Defines a simple U-Net segmentation backbone using segmentation-models-pytorch.
import segmentation_models_pytorch as smp

def unet(num_classes, pretrained):
    weights = "imagenet" if pretrained else None
    return smp.Unet(encoder_name="resnet34", encoder_weights=weights, in_channels=3, classes=num_classes)