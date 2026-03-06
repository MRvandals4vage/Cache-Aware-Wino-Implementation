import torch
import torch.nn as nn
from torchvision import models

def get_resnet18_full():
    """Returns the standard ResNet-18 model (ImageNet version)."""
    # This is the "full" model with 7x7 conv1 and maxpool
    model = models.resnet18(weights=None)
    return model

def get_resnet18_cifar100():
    """Returns a ResNet-18 model modified for CIFAR-100 (32x32 inputs)."""
    model = models.resnet18(weights=None)
    # Modify for CIFAR-100
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

def get_mobilenet_v2_full():
    """Returns the standard MobileNetV2 model (ImageNet version)."""
    model = models.mobilenet_v2(weights=None)
    return model

if __name__ == "__main__":
    # Test ResNet
    model_rn = get_resnet18_full()
    print("Full ResNet-18 conv1:", model_rn.conv1)
    
    # Test MobileNet
    model_mn = get_mobilenet_v2_full()
    print("Full MobileNetV2 features[0]:", model_mn.features[0])
    
    dummy_input = torch.randn(1, 3, 224, 224)
    out_rn = model_rn(dummy_input)
    out_mn = model_mn(dummy_input)
    print(f"ResNet Output shape: {out_rn.shape}")
    print(f"MobileNet Output shape: {out_mn.shape}")
