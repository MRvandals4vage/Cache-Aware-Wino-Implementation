import os
import subprocess
import ctypes
import numpy as np

class FusedWinogradKernel:
    """
    Fused transform, multiplication, and inverse transform.
    Reduces temporary storage and memory traffic by keeping 
    transformed input tiles resident in L1.
    """
    def __init__(self, use_c_ext=True):
        self.use_c_ext = use_c_ext
        self.c_lib = None
        if self.use_c_ext:
            self._compile_and_load_c_ext()

    def _compile_and_load_c_ext(self):
        # Paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        c_src = os.path.join(base_dir, "fused_winograd.c")
        so_file = os.path.join(base_dir, "fused_winograd.so")
        
        # Try to compile if so doesn't exist
        if not os.path.exists(so_file):
            try:
                subprocess.run(
                    ["gcc", "-shared", "-o", so_file, "-fPIC", "-O3", "-march=native", c_src],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except subprocess.CalledProcessError:
                # Compile failed, maybe no gcc
                pass
        
        if os.path.exists(so_file):
            try:
                self.c_lib = ctypes.CDLL(so_file)
                self.c_lib.fused_winograd_f23.argtypes = [
                    np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, flags='C_CONTIGUOUS'),
                    np.ctypeslib.ndpointer(dtype=np.float32, ndim=4, flags='C_CONTIGUOUS'),
                    np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, flags='C_CONTIGUOUS'),
                    ctypes.c_int, ctypes.c_int
                ]
                self.c_lib.fused_winograd_f23.restype = None
            except Exception:
                self.c_lib = None

    def _fallback_numpy(self, input_tile, U):
        """
        Numpy vectorized fallback for F(2,3)
        input_tile: (c_in, 4, 4)
        U: (c_out, c_in, 4, 4)
        Returns output_tile: (c_out, 2, 2)
        """
        BT = np.array([
            [ 1,  0, -1,  0],
            [ 0,  1,  1,  0],
            [ 0, -1,  1,  0],
            [ 0,  1,  0, -1]
        ], dtype=np.float32)
        
        AT = np.array([
            [1,  1,  1,  0],
            [0,  1, -1, -1]
        ], dtype=np.float32)

        # 1. Input Transform
        V = np.matmul(np.matmul(BT, input_tile), BT.T) # shape: (c_in, 4, 4)
        
        # 2. Element-wise multiply & sum over c_in
        # V is (c_in, 4, 4) -> broadcast to (c_out, c_in, 4, 4)
        M = U * V[np.newaxis, ...]
        M_sum = np.sum(M, axis=1) # shape: (c_out, 4, 4)
        
        # 3. Output Transform
        Y = np.matmul(np.matmul(AT, M_sum), AT.T) # shape: (c_out, 2, 2)
        
        return Y

    def execute(self, input_tile, U):
        c_in = input_tile.shape[0]
        c_out = U.shape[0]
        
        input_tile_f32 = np.ascontiguousarray(input_tile, dtype=np.float32)
        U_f32 = np.ascontiguousarray(U, dtype=np.float32)
        
        if self.c_lib is not None:
            output = np.zeros((c_out, 2, 2), dtype=np.float32)
            self.c_lib.fused_winograd_f23(input_tile_f32, U_f32, output, c_in, c_out)
            return output
            
        return self._fallback_numpy(input_tile_f32, U_f32)

if __name__ == "__main__":
    c_in, c_out = 16, 32
    inp = np.random.randn(c_in, 4, 4).astype(np.float32)
    U = np.random.randn(c_out, c_in, 4, 4).astype(np.float32)
    
    kernel = FusedWinogradKernel()
    out = kernel.execute(inp, U)
    print("Fused Kernel Executed, output shape:", out.shape)
