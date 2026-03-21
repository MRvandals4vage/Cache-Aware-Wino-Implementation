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
        Estimate memory working set size of the tile in bytes.
        Considers transformed input tile, transformed kernel tile, 
        accumulation tile, and local thread buffers for a channel-group processing.
        Assuming single thread working set per tile.
        """
        # Transformed input tile: tile_dim x tile_dim x c_in
        input_tile_bytes = tile_dim * tile_dim * c_in * self.dtype_size
        
        # Transformed kernel tile: tile_dim x tile_dim x c_in x c_out (can be large, so we assume blocking over c_out)
        # Assuming block size over c_out is min(c_out, 16) for typical L1 blocking.
        c_out_block = min(c_out, 16)
        kernel_tile_bytes = tile_dim * tile_dim * c_in * c_out_block * self.dtype_size
        
        # Accumulators / output tile
        out_tile_bytes = tile_dim * tile_dim * c_out_block * self.dtype_size
        
        # Thread local temporary buffers (estimated as 2x tile size matrices for intermediate transforms)
        temp_buffer_bytes = 2 * tile_dim * tile_dim * self.dtype_size
        
        total_bytes = input_tile_bytes + kernel_tile_bytes + out_tile_bytes + temp_buffer_bytes
        return total_bytes

    def _compute_reuse_score(self, tile_dim, c_in, c_out):
        """
        Calculate a reuse score to prefer tile shapes that maximize data reuse.
        reuse_score(r) = (C_out * tile_volume(r)) / cache_lines_used(r)
        We define tile_volume as m*m. 
        """
        m = tile_dim - 2 # since r=3
        out_pixels = m * m
        # Number of loaded elements roughly (tile_dim*tile_dim*c_in). cache lines = total_bytes / line_size
        total_bytes = self._estimate_working_set(tile_dim, c_in, min(c_out, 16))
        cache_lines_used = math.ceil(total_bytes / self.line_size)
        
        # Higher is better: more output pixels generated per cache line loaded
        return (c_out * out_pixels) / float(cache_lines_used)

    def select_best_tile(self, c_in, c_out):
        """
        Evaluate candidate tile configurations and reject those that exceed safe capacity.
        Returns the best configuration.
        """
        best_cand = None
        best_score = -1.0
        safe_capacity = self.alpha * self.l1_capacity
        
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
