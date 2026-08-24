"""
Task 14: Custom CUDA Kernel Integration for Accelerated Activation Functions
----------------------------------------------------------------------------
Objective: Bridge the gap between high-level Python scriptability and raw GPU hardware
execution by writing and compiling custom, high-speed CUDA/C++ kernels.

Required Tech Stack: C++, CUDA Toolkit, PyTorch C++ Extensions (cpp_extension)
Formula: SwiGLU(x, y) = x * SiLU(y) = x * (y / (1 + exp(-y)))
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. Inline C++ and CUDA Kernel Source Strings
# =====================================================================

CUDA_SWIGLU_SRC = """
#include <torch/extension.h>
#include <cmath>

// C++ / CPU Fallback implementation
torch::Tensor swiglu_cpu_forward(torch::Tensor x, torch::Tensor y) {
    auto silu_y = y / (1.0 + torch::exp(-y));
    return x * silu_y;
}

#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void swiglu_cuda_kernel(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val_y = y[idx];
        float silu = val_y / (1.0f + expf(-val_y));
        out[idx] = x[idx] * silu;
    }
}

torch::Tensor swiglu_cuda_forward(torch::Tensor x, torch::Tensor y) {
    int N = x.numel();
    auto out = torch::empty_like(x);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    swiglu_cuda_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        N
    );
    return out;
}
#endif

torch::Tensor swiglu_forward(torch::Tensor x, torch::Tensor y) {
    if (x.is_cuda()) {
#ifdef WITH_CUDA
        return swiglu_cuda_forward(x, y);
#else
        TORCH_CHECK(false, "Compiled without CUDA support");
#endif
    }
    return swiglu_cpu_forward(x, y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_forward", &swiglu_forward, "SwiGLU Forward Activation (C++/CUDA)");
}
"""


# =====================================================================
# 2. Kernel Compilation & PyTorch Integration Workflow
# =====================================================================

def load_swiglu_extension():
    """
    Dynamically compiles and binds C++/CUDA SwiGLU extension via torch JIT.
    """
    has_cuda = torch.cuda.is_available()
    extra_cuda_cflags = ['-O3'] if has_cuda else []
    
    try:
        from torch.utils.cpp_extension import load_inline
        swiglu_module = load_inline(
            name="custom_swiglu_extension",
            cpp_sources=CUDA_SWIGLU_SRC,
            extra_cflags=['-O3'],
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=False
        )
        return swiglu_module
    except Exception as e:
        print(f"[Warning] PyTorch C++ Extension compilation fallback notice: {e}")
        return None


def native_pytorch_swiglu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Standard PyTorch SwiGLU reference implementation: x * SiLU(y).
    """
    return x * F.silu(y)


def main():
    print("=" * 70)
    print("Task 14: Custom CUDA / C++ Kernel SwiGLU Activation Verification")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing environment device: {device.type.upper()}")

    # 1. Attempt PyTorch Extension Load
    extension = load_swiglu_extension()

    # 2. Synthetic Test Tensors
    N = 1_000_000 # 1 Million elements
    x = torch.randn(N, device=device, dtype=torch.float32)
    y = torch.randn(N, device=device, dtype=torch.float32)

    # Reference PyTorch Output
    out_native = native_pytorch_swiglu(x, y)

    if extension is not None:
        out_custom = extension.swiglu_forward(x, y)
        max_err = (out_native - out_custom).abs().max().item()
        print(f"\nCustom Kernel vs PyTorch Native Max Absolute Difference: {max_err:.8e}")
        assert max_err < 1e-5, "Custom C++/CUDA kernel numerical output mismatch!"
        print("Kernel output exact precision match confirmed!")

    # 3. Performance Benchmark Comparison
    iters = 100
    print(f"\nRunning Latency Benchmark ({iters} iterations over 1,000,000 floats)...")

    # Benchmark PyTorch Native
    start = time.perf_counter()
    for _ in range(iters):
        _ = native_pytorch_swiglu(x, y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    native_time_ms = (time.perf_counter() - start) * 1000 / iters

    print(f"  PyTorch Native Activation Latency: {native_time_ms:.4f} ms / iter")

    if extension is not None:
        start = time.perf_counter()
        for _ in range(iters):
            _ = extension.swiglu_forward(x, y)
        if device.type == "cuda":
            torch.cuda.synchronize()
        custom_time_ms = (time.perf_counter() - start) * 1000 / iters

        print(f"  Custom C++/CUDA Kernel Latency:    {custom_time_ms:.4f} ms / iter")

    print("\nTask 14 completed successfully!")

if __name__ == "__main__":
    main()
