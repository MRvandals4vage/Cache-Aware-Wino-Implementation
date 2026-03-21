import math
from runtime_cache_probe import RuntimeCacheProbe

class CacheAdaptiveAutotiler:
    def __init__(self, alpha=0.7):
        self.probe = RuntimeCacheProbe()
        self.cache_info = self.probe.get_info()
        self.l1_capacity = self.cache_info["l1d_size_bytes"]
        self.line_size = self.cache_info["line_size_bytes"]
        self.alpha = alpha  # Safe fraction of cache
        self.dtype_size = 4 # Float32
        
        # Candidate configurations: (output_tile_size, kernel_size) => (m, r) => tile_size = m + r - 1
        self.candidates = [
            {"name": "F(2,3)", "m": 2, "r": 3, "tile": 4},
            {"name": "F(4,3)", "m": 4, "r": 3, "tile": 6},
            {"name": "F(6,3)", "m": 6, "r": 3, "tile": 8}
        ]

    def _estimate_working_set(self, tile_dim, c_in, c_out):
        """
        working_set(r) = transformed_input_bytes(r) + transformed_kernel_bytes(r) + accumulation_bytes(r) + thread_local_overhead
        """
        # Transformed input tile: tile_dim x tile_dim x c_in
        transformed_input_bytes = tile_dim * tile_dim * c_in * self.dtype_size
        
        c_out_block = min(c_out, 16)
        transformed_kernel_bytes = tile_dim * tile_dim * c_in * c_out_block * self.dtype_size
        
        # Accumulators / output tile
        accumulation_bytes = tile_dim * tile_dim * c_out_block * self.dtype_size
        
        # Thread local temporary buffers
        thread_local_overhead = 2 * tile_dim * tile_dim * self.dtype_size
        
        working_set_bytes = transformed_input_bytes + transformed_kernel_bytes + accumulation_bytes + thread_local_overhead
        return working_set_bytes

    def _compute_reuse_score(self, tile_dim, c_in, c_out):
        """
        reuse_score(r) = useful_output_work(r) / estimated_cache_lines_touched(r)
        useful_output_work(r) is output pixels generated per channel.
        """
        m = tile_dim - 2 # since r=3
        useful_output_work = c_out * (m * m)
        total_bytes = self._estimate_working_set(tile_dim, c_in, min(c_out, 16))
        estimated_cache_lines_touched = math.ceil(total_bytes / float(self.line_size))
        
        return useful_output_work / estimated_cache_lines_touched

    def select_best_tile(self, c_in, c_out):
        """
        Evaluate candidate tile options based on formulas:
        working_set(r) = transformed_input_bytes(r) + transformed_kernel_bytes(r) + accumulation_bytes(r) + thread_local_overhead
        valid if working_set(r) <= alpha * target_cache_bytes
        """
        best_cand = None
        best_score = -1.0
        target_cache_bytes = self.l1_capacity
        safe_capacity = self.alpha * target_cache_bytes

        
        for cand in self.candidates:
            tile_dim = cand["tile"]
            working_set_bytes = self._estimate_working_set(tile_dim, c_in, c_out)
            
            # Constraint: working_set(r) <= alpha * cache_capacity
            if working_set_bytes <= safe_capacity:
                score = self._compute_reuse_score(tile_dim, c_in, c_out)
                if score > best_score:
                    best_score = score
                    best_cand = cand
                    
        # Fallback to F(2,3) if all exceed L1
        if best_cand is None:
            best_cand = self.candidates[0]
            best_score = self._compute_reuse_score(best_cand["tile"], c_in, c_out)
            
        # Logging
        # print(f"Selected {best_cand['name']} config (score: {best_score:.2f}) for C_in={c_in}, C_out={c_out}")
        return best_cand

if __name__ == "__main__":
    tiler = CacheAdaptiveAutotiler()
    print("Testing C_in=64, C_out=64:", tiler.select_best_tile(64, 64))
    print("Testing C_in=256, C_out=256:", tiler.select_best_tile(256, 256))
    print("Testing C_in=3, C_out=64:", tiler.select_best_tile(3, 64))
