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
            "dram_accesses": 0
        }
        self.log = logging.getLogger("MemoryScheduler")

    def reset_metrics(self):
        self.metrics = {"macs": 0, "dram_accesses": 0}

    def _log_stats(self, op_name, tile_loads, cache_reuse, transform_reuse, dram):
        """Standard logging for scheduling trace."""
        self.log.info(f"{op_name:<20} | Tiles: {tile_loads:>10} | Cache Reuse: {cache_reuse:>10} | Transform Reuse: {transform_reuse:>10} | DRAM: {int(dram):>12}")

    def baseline_direct_conv(self, input_tensor, kernel):
        """Standard direct convolution. Input: (C_in, H, W), Kernel: (OC, C_pg, KH, KW)"""
        C_in, H, W = input_tensor.shape
        OC, C_pg, KH, KW = kernel.shape
        OH, OW = H - KH + 1, W - KW + 1

        macs = OC * C_pg * OH * OW * KH * KW
        dram = (OC * C_pg * KH * KW) + (OC * C_pg * H * W)
        
        self.metrics["macs"] += macs
        self.metrics["dram_accesses"] += dram
        
        self._log_stats("Direct_Conv", H*W, 0, 0, dram)
        return np.zeros((OC, OH, OW))

    def naive_winograd(self, input_tensor, kernels):
        """Naive Winograd: Tile-by-tile, repeated per kernel (OC)."""
        C_in, H, W = input_tensor.shape
        OC, C_pg, KH, KW = kernels.shape
        OH, OW = H - 2, W - 2
        num_tiles_h = (OH + 1) // 2
        num_tiles_w = (OW + 1) // 2
        
        tiles = num_tiles_h * num_tiles_w
        macs = OC * C_pg * tiles * 16
        # DRAM: Weights + Inputs (Redundant load per OC)
        dram = (OC * C_pg * 9) + (OC * C_pg * tiles * 16)
        
        self.metrics["macs"] += macs
        self.metrics["dram_accesses"] += dram
        
        self._log_stats("Naive_Winograd", tiles * OC, 0, 0, dram)
        return np.zeros((OC, OH, OW))

    def cache_aware_winograd(self, input_tensor, kernels):
        """Cache-Aware Winograd: Load input tile once, process all kernels (OC)."""
        C_in, H, W = input_tensor.shape
        OC, C_pg, KH, KW = kernels.shape
        num_tiles_h = (H - 2 + 1) // 2
        num_tiles_w = (W - 2 + 1) // 2
        tiles = num_tiles_h * num_tiles_w

        macs = OC * C_pg * tiles * 16
        # DRAM: Weights once, Input tiles once (per layer)
        dram = (OC * C_pg * 16) + (C_in * tiles * 16)
        
        self.metrics["macs"] += macs
        self.metrics["dram_accesses"] += dram
        
        self._log_stats("Cache_Aware_Wino", tiles, OC, 0, dram)
        return np.zeros((OC, H - 2, W - 2))

    def memory_optimized_winograd(self, input_tensor, kernels):
        """Memory-Optimized Winograd: Channel fusion, transform reuse, minimal writes."""
        C_in, H, W = input_tensor.shape
        OC, C_pg, KH, KW = kernels.shape
        num_tiles_h = (H - 2 + 1) // 2
        num_tiles_w = (W - 2 + 1) // 2
        tiles = num_tiles_h * num_tiles_w

        macs = OC * C_pg * tiles * 16
        # DRAM: Weights (transformed & reused), Input once, Output once
        dram = (OC * C_pg * 16) + (C_in * H * W) + (OC * (H - 2) * (W - 2))
        
        self.metrics["macs"] += macs
        self.metrics["dram_accesses"] += dram
        
        self._log_stats("Mem_Optimized_Wino", tiles, OC, "High", dram)
        return np.zeros((OC, H - 2, W - 2))
