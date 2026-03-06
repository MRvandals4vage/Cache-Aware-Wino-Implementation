import numpy as np
import torch

class MemoryTracer:
    """
    Estimates actual memory traffic (DRAM accesses) based on model architecture.
    Calculates weights, feature map reads, and feature map writes.
    """
    
    @staticmethod
    def estimate_layer_traffic(c_in, c_out, h_in, w_in, k, stride, groups=1):
        """
        Estimates DRAM bytes for a single convolution layer.
        """
        h_out = (h_in - k) // stride + 1
        w_out = (w_in - k) // stride + 1
        
        # Bytes (Assuming float32 = 4 bytes)
        # 1. Weights Read
        weights_bytes = c_out * (c_in // groups) * k * k * 4
        
        # 2. Input Feature Map Read
        # For a standard baseline, we assume we might read it multiple times if OC is large,
        # but modern frameworks cache it. We'll use the scale OC for naive or 1 for optimized.
        # Actually, for research grade, we separate:
        input_fm_bytes = c_in * h_in * w_in * 4
        
        # 3. Output Feature Map Write
        output_fm_bytes = c_out * h_out * w_out * 4
        
        return {
            "weights": weights_bytes,
            "input": input_fm_bytes,
            "output": output_fm_bytes
        }

    @staticmethod
    def estimate_model_traffic(model, input_size=(1, 3, 224, 224), mode="Baseline"):
        """
        Iterates through model and sums up total traffic.
        """
        total_dram_bytes = 0
        
        # We need shapes. Use a simple tracer or hooks.
        input_shapes = {}
        def hook(name):
            def f(module, input, output):
                input_shapes[name] = input[0].shape
            return f

        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                hooks.append(module.register_forward_hook(hook(name)))

        # Run forward to get shapes
        with torch.no_grad():
            dummy = torch.randn(*input_size)
            _ = model(dummy)
        for h in hooks: h.remove()

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                c_in = module.in_channels
                c_out = module.out_channels
                k = module.kernel_size[0]
                stride = module.stride[0]
                groups = module.groups
                
                shape = input_shapes[name]
                h_in, w_in = shape[2], shape[3]
                
                traffic = MemoryTracer.estimate_layer_traffic(c_in, c_out, h_in, w_in, k, stride, groups)
                
                # Apply mode-specific multipliers
                if mode == "Baseline":
                    # Naive reuse might cause multiple input reads
                    total_dram_bytes += traffic["weights"] + (traffic["input"] * 1.5) + traffic["output"]
                else: # Optimized (TVM/Winograd)
                    # Minimized writes/reads
                    total_dram_bytes += traffic["weights"] + traffic["input"] + traffic["output"]
        
        # Convert bytes to "DRAM Accesses" (Assuming 4-byte words as requested by previous steps)
        return total_dram_bytes / 4

if __name__ == "__main__":
    from cnn_model import get_resnet18_full
    model = get_resnet18_full()
    tracer = MemoryTracer()
    accesses = tracer.estimate_model_traffic(model)
    print(f"Estimated DRAM Accesses: {int(accesses):,}")
