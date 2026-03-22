import math
import os
import json
from src.runtime_cache_probe import build_platform_descriptor

class CacheAdaptiveAutotiler:
    def __init__(self, platform_descriptor=None, alpha=0.7):
        if platform_descriptor is None:
            self.cache_info = build_platform_descriptor()
        else:
            self.cache_info = platform_descriptor
            
        self.l1_capacity = self.cache_info.get("l1d_size_bytes", 32768)
        self.line_size = self.cache_info.get("line_size_bytes", 64)
        self.alpha = alpha  # Safe fraction of cache
        self.dtype_size = 4 # Float32
        
        # Candidate configurations: (output_tile_size, kernel_size) => (m, r) => tile_size = m + r - 1
        self.candidates = [
            {"name": "F(2,3)", "m": 2, "r": 3, "tile": 4},
            {"name": "F(4,3)", "m": 4, "r": 3, "tile": 6},
            {"name": "F(6,3)", "m": 6, "r": 3, "tile": 8}
        ]

    def compute_working_set(self, tile_dim, c_in, c_out):
        """
        working_set(r) = transformed_input_bytes(r) + transformed_kernel_bytes(r) + accumulation_bytes(r) + thread_local_overhead
        """
        transformed_input_bytes = tile_dim * tile_dim * c_in * self.dtype_size
        c_out_block = min(c_out, 16)
        transformed_kernel_bytes = tile_dim * tile_dim * c_in * c_out_block * self.dtype_size
        accumulation_bytes = tile_dim * tile_dim * c_out_block * self.dtype_size
        thread_local_overhead = 2 * tile_dim * tile_dim * self.dtype_size
        
        working_set_bytes = transformed_input_bytes + transformed_kernel_bytes + accumulation_bytes + thread_local_overhead
        return working_set_bytes

    def compute_reuse_score(self, tile_dim, c_in, c_out):
        """
        reuse_score(r) = useful_output_work(r) / estimated_cache_lines_touched(r)
        useful_output_work(r) is output pixels generated per channel.
        """
        m = tile_dim - 2 # assuming r=3
        useful_output_work = c_out * (m * m)
        total_bytes = self.compute_working_set(tile_dim, c_in, min(c_out, 16))
        estimated_cache_lines_touched = math.ceil(total_bytes / float(self.line_size))
        
        return useful_output_work / estimated_cache_lines_touched

    def select_best_tile(self, c_in, c_out):
        """
        Evaluate candidate tile options based on working set constraints and reuse scores.
        """
        best_cand = None
        best_score = -1.0
        
        target_cache_bytes = self.l1_capacity
        safe_capacity = self.alpha * target_cache_bytes
        
        for cand in self.candidates:
            tile_dim = cand["tile"]
            working_set_bytes = self.compute_working_set(tile_dim, c_in, c_out)
            
            if working_set_bytes <= safe_capacity:
                score = self.compute_reuse_score(tile_dim, c_in, c_out)
                if score > best_score:
                    best_score = score
                    best_cand = cand
                    
        # Fallback behaviour: default to F(2,3)
        if best_cand is None:
            best_cand = self.candidates[0]
            best_score = self.compute_reuse_score(best_cand["tile"], c_in, c_out)
            
        return {
            "selected_tile": best_cand,
            "c_in": c_in,
            "c_out": c_out,
            "score": best_score,
            "l1_capacity": self.l1_capacity,
            "working_set": self.compute_working_set(best_cand["tile"], c_in, c_out)
        }

    def save_autotiling_decision(self, decision, path="artifacts/autotiling_decisions.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # load existing if exists
        decisions = []
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    decisions = json.load(f)
            except Exception:
                pass
                
        decisions.append(decision)
        
        with open(path, "w") as f:
            json.dump(decisions, f, indent=2)

if __name__ == "__main__":
    tiler = CacheAdaptiveAutotiler()
    dec = tiler.select_best_tile(64, 64)
    print("Decision:", dec)
    tiler.save_autotiling_decision(dec)
