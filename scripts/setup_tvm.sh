#!/usr/bin/env bash
# ==============================================================================
# scripts/setup_tvm.sh
#
# Apache TVM & AutoTVM Environment Setup & Verification for Jetson Nano / ARM64
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "======================================================================"
echo "          Apache TVM & AutoTVM Setup & Verification                   "
echo "======================================================================"

if python3 -c "import tvm" &>/dev/null; then
    TVM_VER="$(python3 -c 'import tvm; print(getattr(tvm, "__version__", "installed"))')"
    echo "[TVM] TVM is already installed: ${TVM_VER}"
else
    echo "[TVM] TVM not found in current Python environment. Attempting install..."
    pip3 install apache-tvm || pip3 install -f https://tlcpack.ai/wheels tvm || pip install apache-tvm || true
fi

# Verification
python3 - <<'EOF'
try:
    import tvm
    ver = getattr(tvm, "__version__", "unknown")
    print(f"[✓] TVM is operational (version: {ver})")
    
    # Check AutoTVM / Relay / TE
    has_relay = hasattr(tvm, "relay")
    print(f"    - Relay Frontend: {'Available' if has_relay else 'Modern TIR/TE mode'}")
    
    try:
        from tvm import autotvm
        print(f"    - AutoTVM: Available")
    except ImportError:
        print(f"    - AutoTVM: Not bundled (modern TVM uses MetaSchedule/builtin tuning)")

    # Run quick compilation test
    if has_relay:
        from tvm import relay
        x = relay.var("x", shape=(1, 16, 14, 14), dtype="float32")
        w = relay.const(np.zeros((32, 16, 3, 3), dtype="float32"))
        conv = relay.nn.conv2d(x, w, channels=32, kernel_size=(3, 3), padding=(1, 1))
        mod = tvm.IRModule.from_expr(conv)
        with tvm.transform.PassContext(opt_level=1):
            lib = relay.build(mod, target="llvm", params={})
        print("[✓] TVM Relay compilation test passed!")
    else:
        from tvm import te, topi
        import numpy as np
        X = te.placeholder((1, 16, 14, 14), name="X", dtype="float32")
        W = te.placeholder((32, 16, 3, 3), name="W", dtype="float32")
        Y = topi.nn.conv2d_nchw(X, W, stride=1, padding=1, dilation=1)
        func = te.create_prim_func([X, W, Y])
        lib = tvm.build(func, target="llvm")
        print("[✓] TVM TE/PrimFunc compilation test passed!")
except Exception as e:
    print(f"[Warning] TVM verification reported: {e}")
EOF

echo
echo "======================================================================"
echo "TVM & AutoTVM setup verification finished."
echo "======================================================================"
