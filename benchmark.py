import time
import torch  # type: ignore
import numpy as np  # type: ignore
from cnn_model import get_resnet18_full, get_vgg16_full, get_alexnet_full, get_resnet34_full  # type: ignore
from memory_scheduler import MemoryScheduler  # type: ignore
from energy_model import EnergyModel  # type: ignore

class BenchmarkRunner:
    """Class to run benchmarks on a CNN model with different strategies."""

    def __init__(self, mode="Baseline", model_name="resnet18"):
        self.mode = mode
        self.model_name = model_name
        self.scheduler = MemoryScheduler(mode=mode)
        self.energy_model = EnergyModel()
        self.device = torch.device("cpu")
        
        if model_name == "resnet18":
            self.model = get_resnet18_full().to(self.device).eval()
            self.input_size = (1, 3, 224, 224)
        elif model_name == "vgg16":
            self.model = get_vgg16_full().to(self.device).eval()
            self.input_size = (1, 3, 224, 224)
        elif model_name == "alexnet":
            self.model = get_alexnet_full().to(self.device).eval()
            self.input_size = (1, 3, 224, 224)
        elif model_name == "resnet34":
            self.model = get_resnet34_full().to(self.device).eval()
            self.input_size = (1, 3, 224, 224)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        self.dummy_input = torch.randn(*self.input_size).to(self.device)

    def run(self, average_power_mw=0):
        """Perform measurement-driven inference profiling."""
        
        # 1. Tracing Algorithm MACs and Memory Traffic
        self.scheduler.reset_metrics()
        
        # We need shapes. Use a simple tracer or hooks.
        input_shapes = {}
        def hook(name):
            def f(module, input, output):
                input_shapes[name] = input[0].shape
            return f

        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                hooks.append(module.register_forward_hook(hook(name)))  # type: ignore

        # Run one forward pass to collect input shapes
        with torch.no_grad():
            _ = self.model(self.dummy_input)

        for h in hooks:
            h.remove()

        # Iterate and calculate algorithmic metrics
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                c_in = module.in_channels  # type: ignore
                c_out = module.out_channels  # type: ignore
                k = module.kernel_size[0]  # type: ignore
                stride = module.stride[0]  # type: ignore
                
                shape = input_shapes[name]
                h_in, w_in = shape[2], shape[3]
                
                # Winograd applies ONLY to kernel_size 3x3 with stride 1 (Standard research scope)
                if k == 3 and stride == 1:
                    if self.mode == "Baseline":
                        self.scheduler.baseline_direct_conv(c_in, c_out, h_in, w_in, k, stride)
                    elif self.mode == "Naive Winograd":
                        self.scheduler.winograd_f23(c_in, c_out, h_in, w_in, mode="Naive")
                    elif self.mode == "Cache-Aware":
                        self.scheduler.winograd_f23(c_in, c_out, h_in, w_in, mode="Cache-Aware")
                    elif self.mode == "TVM Model":
                        self.scheduler.winograd_f23(c_in, c_out, h_in, w_in, mode="Optimized")
                else:
                    # Non 3x3 layers or strided layers remain baseline
                    self.scheduler.baseline_direct_conv(c_in, c_out, h_in, w_in, k, stride)

        # 2. Measure Latency using high-precision timers
        # Warmup
        for _ in range(20):
            with torch.no_grad():
                _ = self.model(self.dummy_input)
        
        # Measurement
        start_time = time.perf_counter()
        num_runs = 100
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(self.dummy_input)
        end_time = time.perf_counter()
        
        avg_latency_s = (end_time - start_time) / num_runs
        avg_latency_ms = avg_latency_s * 1000.0
        
        # 3. Final Calculations (Measurement + Algorithm)
        total_macs = self.scheduler.metrics["macs"]
        total_bytes = self.scheduler.metrics["bytes_transferred"]
        energy_mj = self.energy_model.calculate_energy(average_power_mw, avg_latency_ms)
        efficiency = self.energy_model.calculate_efficiency(total_macs, energy_mj)
        
        return {
            "Strategy": self.mode,
            "Model": self.model_name,
            "time_ms": avg_latency_ms,
            "throughput_fps": 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0,
            "MACs": total_macs,
            "Bytes": total_bytes,
            "Energy (mJ)": energy_mj,
            "efficiency": efficiency,
            "average_power_mw": average_power_mw
        }
