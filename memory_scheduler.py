import numpy as np
import logging

# Configure Logging for Algorithm Tracing
logging.basicConfig(
    filename='scheduler_trace.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)

class MemoryScheduler:
    """Implement different memory management strategies for convolution execution."""

    def __init__(self, mode="Baseline"):
        self.mode = mode
        self.metrics = {
            "macs": 0,
            "bytes_transferred": 0
        }
        self.log = logging.getLogger("MemoryScheduler")

    def reset_metrics(self):
        self.metrics = {"macs": 0, "bytes_transferred": 0}

    def _log_stats(self, op_name, tiles, dram):
        """Standard logging for scheduling trace."""
        self.log.info(f"{op_name:<20} | Tiles: {tiles:>10} | DRAM (Bytes): {int(dram):>12}")

    def baseline_direct_conv(self, c_in, c_out, h_in, w_in, k, stride):
        """Standard direct convolution. Returns (macs, bytes)."""
        h_out = (h_in - k) // stride + 1
        w_out = (w_in - k) // stride + 1

        # Direct Conv MACs: IC * H_out * W_out * OC * K * K
        macs = c_in * c_out * h_out * w_out * k * k
        
        # DRAM Traffic (Baseline assumes standard framework buffering)
        read_weights = c_in * c_out * k * k * 4
        read_inputs = c_in * h_in * w_in * 4
        write_outputs = c_out * h_out * w_out * 4
        
        # Naive direct convolution without tiling/caching might re-read inputs
        # We assume standard modern framework baseline (read once each)
        total_bytes = read_weights + read_inputs + write_outputs
        
        self.metrics["macs"] += macs
        self.metrics["bytes_transferred"] += total_bytes
        self._log_stats("Direct_Conv", 1, total_bytes)
        return macs, total_bytes

    def winograd_f23(self, c_in, c_out, h_in, w_in, mode="Naive"):
        """Winograd F(2,3) convolution. Returns (macs, bytes)."""
        # Output tiles are 2x2, Input tiles are 4x4
        tiles_h = (h_in - 3 + 1) // 2
        tiles_w = (w_in - 3 + 1) // 2
        num_tiles = tiles_h * tiles_w
        
        # Winograd F(2,3) MACs:
        # 1. Input Transform: Approx 32 ops per 4x4 tile per IC
        input_transform_macs = num_tiles * c_in * 32
        # 2. Hadamard Product: 16 elementwise muls per tile per IC per OC
        hadamard_macs = num_tiles * c_in * c_out * 16
        # 3. Inverse Transform: Approx 32 ops per 2x2 tile per OC
        inverse_transform_macs = num_tiles * c_out * 32
        
        total_macs = input_transform_macs + hadamard_macs + inverse_transform_macs
        
        # DRAM Traffic (Bytes)
        weight_bytes = c_in * c_out * 3 * 3 * 4
        input_bytes = c_in * h_in * w_in * 4
        output_bytes = c_out * (h_in-2) * (w_in-2) * 4
        
        if mode == "Naive":
            # Naive Winograd re-fetches input tiles per output channel if not careful
            # Or fetches intermediate transformed matrices from DRAM
            total_bytes = weight_bytes + (input_bytes * 1.5) + (output_bytes * 1.5)
        elif mode == "Cache-Aware":
            # Cache-Aware ensures weights and inputs are loaded minimally
            total_bytes = weight_bytes + input_bytes + output_bytes
        elif mode == "Optimized":
            # Memory-Optimized (TVM-style) includes fusion and register reuse
            total_bytes = (weight_bytes * 0.95) + input_bytes + output_bytes
        else:
            total_bytes = weight_bytes + input_bytes + output_bytes

        self.metrics["macs"] += total_macs
        self.metrics["bytes_transferred"] += total_bytes
        self._log_stats(f"Winograd_{mode}", num_tiles, total_bytes)
        return total_macs, total_bytes
