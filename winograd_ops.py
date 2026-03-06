import numpy as np

class WinogradOps:
    """Core Winograd transformations for F(2x2, 3x3)."""
    
    # Transformation matrices for F(2x2, 3x3)
    G = np.array([
        [1,      0,    0],
        [0.5,  0.5,  0.5],
        [0.5, -0.5,  0.5],
        [0,      0,    1]
    ])
    
    BT = np.array([
        [ 1,  0, -1,  0],
        [ 0,  1,  1,  0],
        [ 0, -1,  1,  0],
        [ 0,  1,  0, -1]
    ])
    
    AT = np.array([
        [1,  1,  1,  0],
        [0,  1, -1, -1]
    ])

    @staticmethod
    def kernel_transform(g):
        """Transform 3x3 kernel g into 4x4 Winograd space."""
        # U = G * g * GT
        return WinogradOps.G @ g @ WinogradOps.G.T

    @staticmethod
    def input_transform(d):
        """Transform 4x4 input tile d into 4x4 Winograd space."""
        # V = BT * d * B
        # Since B is BT.T, we can use BT.T
        return WinogradOps.BT @ d @ WinogradOps.BT.T

    @staticmethod
    def output_transform(m):
        """Transform 4x4 element-wise product m into 2x2 output space."""
        # Y = AT * m * A
        # Since A is AT.T, we can use AT.T
        return WinogradOps.AT @ m @ WinogradOps.AT.T

if __name__ == "__main__":
    w_ops = WinogradOps()
    kernel = np.random.randn(3, 3)
    tile = np.random.randn(4, 4)
    
    U = w_ops.kernel_transform(kernel)
    V = w_ops.input_transform(tile)
    M = U * V
    Y = w_ops.output_transform(M)
    
    print(f"Kernel transform shape: {U.shape}")
    print(f"Input transform shape: {V.shape}")
    print(f"Output shape: {Y.shape}")
    
    # Simple validation using direct convolution
    # Direct conv of 4x4 with 3x3 kernel results in 2x2.
    direct = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            direct[i, j] = np.sum(tile[i:i+3, j:j+3] * kernel)
    
    # Difference should be very small
    print(f"Direct vs Winograd Difference sum: {np.abs(direct - Y).sum():.4e}")
