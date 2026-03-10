import time
import torch
import numpy as np
from cnn_model import get_resnet18_cifar100, get_resnet18_full, get_vgg16_full, get_alexnet_full, get_resnet34_full
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
        elif model_name == "vgg16":
            self.model = get_vgg16_full().to(self.device).eval()
            self.input_size = (1, 3, 224, 224)
        elif model_name == "alexnet":
            self.model = get_alexnet_full().to(self.device).eval()
            self.input_size = (1, 3, 224, 224)
        elif model_name == "resnet34":
            self.model = get_resnet34_full().to(self.device).eval()
            self.input_size = (1, 3, 224, 224)
        elif model_name == "resnet18_cifar":
            self.model = get_resnet18_cifar100().to(self.device).eval()
            self.input_size = (1, 3, 32, 32)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        self.dummy_input = torch.randn(*self.input_size).to(self.device)

    def run(self, average_power_mw=0):
        """Perform measurement-driven inference profiling."""
        
        # 1. Capture Dynamic MACs using thop
        from energy_model import compute_dynamic_macs
        total_macs = compute_dynamic_macs(self.model, self.input_size)
        
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
        
        # 3. Dynamic Memory Traffic (from memory_trace)
        from memory_trace import MemoryTracer
        dram_mode = "Optimized" if self.mode in ["Naive Winograd", "Cache-Aware", "TVM Model"] else "Baseline"
        total_dram_accesses = MemoryTracer.estimate_model_traffic(self.model, self.input_size, mode=dram_mode)
        
        # 4. Energy (Derived from Measured Power)
        energy_mj = self.energy_model.calculate_energy(average_power_mw, avg_latency_ms)
        efficiency = self.energy_model.calculate_efficiency(total_macs, energy_mj)
        
        return {
            "Strategy": self.mode,
            "Model": self.model_name,
            "time_ms": avg_latency_ms,
            "throughput_fps": 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0,
            "MACs": total_macs,
            "DRAM": total_dram_accesses,
            "Energy (mJ)": energy_mj,
            "efficiency": efficiency,
            "average_power_mw": average_power_mw
        }
