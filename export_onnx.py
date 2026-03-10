import torch
import os
from cnn_model import get_resnet18_full, get_vgg16_full, get_alexnet_full, get_resnet34_full

def export_models():
    """Exports ResNet-18, VGG16, AlexNet, and ResNet34 to ONNX format."""
    print("Exporting models to ONNX...")
    
    models = {
        "resnet18": get_resnet18_full(),
        "vgg16": get_vgg16_full(),
        "alexnet": get_alexnet_full(),
        "resnet34": get_resnet34_full()
    }
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    for name, model in models.items():
        model.eval()
        onnx_path = f"{name}.onnx"
        
        print(f"  Exporting {name} to {onnx_path}...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"  {name} exported successfully.")

if __name__ == "__main__":
    export_models()
