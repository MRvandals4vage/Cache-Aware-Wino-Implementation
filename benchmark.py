import time
import torch
import numpy as np
from cnn_model import get_resnet18_cifar100, get_resnet18_full
from memory_scheduler import MemoryScheduler
from energy_model import EnergyModel

class BenchmarkRunner:
    """Class to run benchmarks on a CNN model with different strategies."""

    # Approximate Jetson Nano CPU constraints
    PEAK_FLOPS = 10e9 # 10 GFLOPS
    DRAM_BW = 25.6e9  # 25.6 GB/s
    WORD_SIZE = 4     # float32

    def __init__(self, mode="Baseline", model_name="resnet18"):
        self.mode = mode
        self.model_name = model_name
        self.scheduler = MemoryScheduler(mode=mode)
        self.energy_model = EnergyModel()
        self.device = torch.device("cpu")
        
        if model_name == "resnet18":
            self.model = get_resnet18_full().to(self.device).eval()
            self.input_size = (1, 3, 224, 224)
        elif model_name == "resnet18_cifar":
            self.model = get_resnet18_cifar100().to(self.device).eval()
            self.input_size = (1, 3, 32, 32)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        self.dummy_input = torch.randn(*self.input_size).to(self.device)

    def run(self):
        """Perform simulation based on layer configs and selected mode."""
        total_macs = 0
        total_dram = 0
        
        input_shapes = {}
        
        def hook(name):
            def f(module, input, output):
                input_shapes[name] = input[0].shape
            return f

        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                hooks.append(module.register_forward_hook(hook(name)))

        # Run one forward pass to collect input shapes
        with torch.no_grad():
            _ = self.model(self.dummy_input)

        for h in hooks:
            h.remove()

        # Iterate and calculate metrics
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                c_in = module.in_channels
                c_out = module.out_channels
                k = module.kernel_size[0]
                
                shape = input_shapes[name]
                h_in, w_in = shape[2], shape[3]
                
                # Weight tensor for shape information
                weight = module.weight.detach().cpu().numpy()
                
                self.scheduler.reset_metrics()
                
                # Winograd applies ONLY to kernel_size 3x3 with stride 1 (Standard research scope)
                if k == 3 and module.stride == (1, 1):
                    if self.mode == "Baseline":
                        self.scheduler.baseline_direct_conv(np.zeros((c_in, h_in, w_in)), weight)
                    elif self.mode == "Naive Winograd":
                        self.scheduler.naive_winograd(np.zeros((c_in, h_in, w_in)), weight)
                    elif self.mode == "Cache-Aware":
                        self.scheduler.cache_aware_winograd(np.zeros((c_in, h_in, w_in)), weight)
                    elif self.mode == "TVM Model":
                        self.scheduler.memory_optimized_winograd(np.zeros((c_in, h_in, w_in)), weight)
                else:
                    # Non 3x3 layers or strided layers remain baseline
                    self.scheduler.baseline_direct_conv(np.zeros((c_in, h_in, w_in)), weight)

                total_macs += self.scheduler.metrics["macs"]
                total_dram += self.scheduler.metrics["dram_accesses"]

        # Calculate time based on hardware model
        # MAC_Time = 2 * MACs / PeakFLOPS (since 1 MAC = 2 FLOPS)
        comp_time = (2 * total_macs) / self.PEAK_FLOPS
        # DRAM_Time = DRAM_Accesses * WordSize / Bandwidth
        mem_time = (total_dram * self.WORD_SIZE) / self.DRAM_BW
        
        estimated_time_ms = (comp_time + mem_time) * 1000.0
        
        energy_mJ = self.energy_model.calculate_energy(total_macs, total_dram)
        efficiency = self.energy_model.calculate_efficiency(total_macs, energy_mJ)
        
        return {
            "Strategy": self.mode,
            "Model": self.model_name,
            "time_ms": estimated_time_ms,
            "MACs": total_macs,
            "DRAM": total_dram,
            "Total Energy (mJ)": energy_mJ,
            "efficiency": efficiency,
            "dram_mac": total_dram / total_macs if total_macs > 0 else 0
        }
