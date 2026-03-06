import onnx
import tvm
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
import numpy as np
import time
import os

def compile_tvm_model(onnx_model_path, target="llvm"):
    """
    Compiles an ONNX model using TVM Relay and AutoScheduler.
    Target should be "llvm -mtriple=aarch64-linux-gnu -mattr=+neon" for Jetson Nano.
    """
    print(f"  TVM: Loading ONNX model {onnx_model_path}...")
    onnx_model = onnx.load(onnx_model_path)
    
    # Set input shape
    shape_dict = {"input": (1, 3, 224, 224)}
    
    # 1. Convert ONNX to Relay
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    
    # 2. Apply AutoScheduler Tuning (Simplified for benchmarking framework)
    # In a real scenario, we would run a tuning log. 
    # Here we perform a basic optimization pass.
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    print(f"  TVM: Extracted {len(tasks)} scheduler tasks.")

    # 3. Compile optimized kernels
    print(f"  TVM: Compiling with Relayout and Optimization (Target: {target})...")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    
    return lib

def run_tvm_inference(lib, num_iterations=100):
    """
    Runs inference using the compiled TVM module and measures latency.
    """
    dev = tvm.cpu(0)
    module = graph_executor.GraphModule(lib["default"](dev))
    
    # Generate input
    input_data = np.random.randn(1, 3, 224, 224).astype("float32")
    module.set_input("input", input_data)
    
    # Warmup
    print("    TVM: Warming up model (20 iterations)...")
    for _ in range(20):
        module.run()
        
    # Benchmark
    print(f"    TVM: Running benchmarking loop ({num_iterations} iterations)...")
    latencies = []
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        module.run()
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000.0) # ms
        
    avg_latency = np.mean(latencies)
    throughput = 1000.0 / avg_latency
    
    print(f"    TVM Avg Latency: {avg_latency:.2f} ms")
    print(f"    TVM Throughput: {throughput:.2f} FPS")
    
    return {
        "latency_ms": avg_latency,
        "throughput_fps": throughput
    }

if __name__ == "__main__":
    # Test compilation if onnx exists
    if os.path.exists("resnet18.onnx"):
        lib = compile_tvm_model("resnet18.onnx")
        run_tvm_inference(lib)
