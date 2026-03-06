import onnxruntime as ort
import numpy as np
import time

def run_onnx_inference(onnx_model_path, num_iterations=100):
    """
    Runs ONNX Runtime inference using CPU only and measures latency/throughput.
    """
    print(f"  Starting ONNX Runtime benchmarks for {onnx_model_path}...")
    
    # 3. Force CPU Execution
    providers = ['CPUExecutionProvider']
    
    # 1. Load ONNX model
    session = ort.InferenceSession(onnx_model_path, providers=providers)
    
    input_name = session.get_inputs()[0].name
    
    # 2. Generate random input tensor using numpy
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Warmup
    print("    Warming up model...")
    for _ in range(20):
        _ = session.run(None, {input_name: input_data})
        
    # Benchmark loop
    print(f"    Running {num_iterations} iterations...")
    latencies = []
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        _ = session.run(None, {input_name: input_data})
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000.0) # Convert to ms
        
    # 4. Measure metrics
    avg_latency = np.mean(latencies)
    throughput = 1000.0 / avg_latency
    
    print(f"    Avg Latency: {avg_latency:.2f} ms")
    print(f"    Throughput: {throughput:.2f} FPS")
    
    return {
        "latency_ms": avg_latency,
        "throughput_fps": throughput
    }

if __name__ == "__main__":
    # Test if file exists
    import os
    if os.path.exists("resnet18.onnx"):
        run_onnx_inference("resnet18.onnx")
    else:
        print("Run export_onnx.py first.")
