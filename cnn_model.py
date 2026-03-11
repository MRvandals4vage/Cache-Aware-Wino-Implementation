import torch
import torch.nn as nn
from torchvision import models

def get_resnet18_full():
    """Returns the standard ResNet-18 model (ImageNet version)."""
    # Using pretrained=False for compatibility with older torchvision (Jetson Nano)
    model = models.resnet18(pretrained=False)
    return model

def get_vgg16_full():
    """Returns the standard VGG16 model (ImageNet version)."""
    model = models.vgg16(pretrained=False)
    return model

def get_alexnet_full():
    """Returns the standard AlexNet model (ImageNet version)."""
    model = models.alexnet(pretrained=False)
    return model

def get_resnet34_full():
    """Returns the standard ResNet-34 model (ImageNet version)."""
    model = models.resnet34(pretrained=False)
    return model

if __name__ == "__main__":
    # Test architectures
    archs = {
        "ResNet18": get_resnet18_full(),
        "VGG16": get_vgg16_full(),
        "AlexNet": get_alexnet_full(),
        "ResNet34": get_resnet34_full()
    }
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    for name, model in archs.items():
        model.eval()
        out = model(dummy_input)
        print(f"{name} Output shape: {out.shape}")

