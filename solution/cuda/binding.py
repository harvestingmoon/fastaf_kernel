"""
PyTorch Extension Binding for Gated Delta Net Prefill CUDA Kernel.

Optimized for zero-overhead dispatch, TorchScript/Inductor compatibility,
and maximum memory bandwidth utilization.
"""

import os
import hashlib
import torch
from torch.utils.cpp_extension import load_inline

try:
    from tvm_ffi import register_func as _tvm_register_func
except ImportError:
    try:
        from tvm.ffi import register_func as _tvm_register_func
    except ImportError:
        _tvm_register_func = None

def register_func(_name):
    """No-op decorator — TVM-FFI registration is skipped to avoid runtime hangs.
    The kernel is called directly by the benchmarking framework as a Python callable."""
    def deco(fn):
        return fn
    return deco

# Cache for the loaded module
_cuda_module = None

def _get_cuda_source():
    """Read the CUDA kernel source from kernel.cu."""
    kernel_path = os.path.join(os.path.dirname(__file__), 'kernel.cu')
    with open(kernel_path, 'r') as f:
        return f.read()

def _load_cuda_module():
    """Load CUDA kernel using PyTorch's inline compilation."""
    global _cuda_module
    if _cuda_module is not None:
        return _cuda_module
    
    cuda_source = _get_cuda_source()
    
    # Optimized C++ wrapper with PyTorch Stream awareness
    cpp_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" int launch_gated_delta_net_prefill_raw(
    const void*  q,
    const void*  k,
    const void*  v,
    const float* state,
    const float* A_log,
    const void*  a,          // bf16 [T, v_heads]
    const float* dt_bias,
    const void*  b,          // bf16 [T, v_heads]
    const int*   cu_seqlens,
    float        scale,
    void*        output,
    float*       new_state,
    int num_seqs,
    int q_heads,
    int k_heads,
    int v_heads,
    int num_sab_heads,
    int head_size,
    cudaStream_t stream
);

void launch_gated_delta_net_prefill(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor state,
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b,
    torch::Tensor cu_seqlens,
    float scale,
    torch::Tensor output,
    torch::Tensor new_state,
    int num_seqs,
    int q_heads,
    int k_heads,
    int v_heads,
    int num_sab_heads,
    int head_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_gated_delta_net_prefill_raw(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        state.numel() > 0 ? state.data_ptr<float>() : nullptr,
        A_log.data_ptr<float>(),
        a.data_ptr(),          // bf16 void*
        dt_bias.data_ptr<float>(),
        b.data_ptr(),          // bf16 void*
        cu_seqlens.data_ptr<int>(),
        scale,
        output.data_ptr(),
        new_state.data_ptr<float>(),
        num_seqs,
        q_heads,
        k_heads,
        v_heads,
        num_sab_heads,
        head_size,
        stream
    );
}

extern "C" int launch_gdn_decode(
    const void*  q,
    const void*  k,
    const void*  v,
    const float* state,
    const float* A_log,
    const void*  a,
    const float* dt_bias,
    const void*  b,
    float        scale,
    void*        output,
    float*       new_state,
    int batch_size,
    int num_v_heads,
    int num_qk_heads,
    int head_size,
    cudaStream_t stream
);

void launch_gdn_decode_wrapper(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor state,
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b,
    float         scale,
    torch::Tensor output,
    torch::Tensor new_state,
    int batch_size,
    int num_v_heads,
    int num_qk_heads,
    int head_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int err = launch_gdn_decode(
        q.data_ptr(), k.data_ptr(), v.data_ptr(),
        state.numel() > 0 ? state.data_ptr<float>() : nullptr,
        A_log.data_ptr<float>(),
        a.data_ptr(), dt_bias.data_ptr<float>(), b.data_ptr(),
        scale,
        output.data_ptr(),
        new_state.data_ptr<float>(),
        batch_size, num_v_heads, num_qk_heads, head_size,
        stream
    );
    TORCH_CHECK(err == 0, "launch_gdn_decode failed with CUDA error ", err);
}
"""
    
    source_hash = hashlib.sha1((cpp_source + cuda_source).encode("utf-8")).hexdigest()[:12]
    ext_name = f"gdn_kernel_{source_hash}"

    _cuda_module = load_inline(
        name=ext_name,
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['launch_gated_delta_net_prefill', 'launch_gdn_decode_wrapper'],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math', '-Xptxas', '-O3'], # Maximize PTX optimization
        verbose=False,
        with_cuda=True
    )
    
    return _cuda_module

@register_func("kernel")
@register_func("flashinfer.kernel")
@torch.compiler.allow_in_graph # Crucial for torch.compile() fusion
def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    module = _load_cuda_module()
    
    total_seq_len, orig_q_heads, head_size = q.shape
    _, orig_k_heads, _ = k.shape
    _, orig_v_heads, _ = v.shape
    
    num_sab_heads = max(orig_q_heads, orig_k_heads, orig_v_heads)
    num_seqs = cu_seqlens.shape[0] - 1
    
    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32, copy=False)
    
    if state is None:
        state = torch.empty(0, device=q.device, dtype=torch.float32)

    # Output layout: [T, num_sab_heads, D]
    output    = torch.empty((total_seq_len, num_sab_heads, head_size), dtype=q.dtype, device=q.device)
    new_state = torch.empty((num_seqs, num_sab_heads, head_size, head_size), dtype=torch.float32, device=q.device)
    
    module.launch_gated_delta_net_prefill(
        q.contiguous(), k.contiguous(), v.contiguous(), state.contiguous(),
        A_log.contiguous(), a.contiguous(), dt_bias.contiguous(), b.contiguous(), 
        cu_seqlens.contiguous(), float(scale),
        output, new_state,
        int(num_seqs), int(orig_q_heads), int(orig_k_heads), int(orig_v_heads),
        int(num_sab_heads), int(head_size)
    )
    
    return output, new_state


@register_func("kernel_decode")
@register_func("flashinfer.kernel_decode")
@torch.compiler.allow_in_graph
def kernel_decode(q, k, v, state, A_log, a, dt_bias, b, scale):
    """Decode kernel: q/k/v/a/b in [B,1,H,D] / [B,1,Hv] layout (seq_len=1 squeezed internally)."""
    module = _load_cuda_module()

    batch_size, _, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[2]

    # Squeeze the seq_len=1 dimension expected by the framework
    q = q.squeeze(1)   # [B, Hq, D]
    k = k.squeeze(1)   # [B, Hk, D]
    v = v.squeeze(1)   # [B, Hv, D]
    a = a.squeeze(1)   # [B, Hv]
    b = b.squeeze(1)   # [B, Hv]

    if state is None:
        state = torch.empty(0, device=q.device, dtype=torch.float32)

    output    = torch.empty((batch_size, num_v_heads, head_size),
                            dtype=torch.bfloat16, device=q.device)
    new_state = torch.empty((batch_size, num_v_heads, head_size, head_size),
                            dtype=torch.float32, device=q.device)

    module.launch_gdn_decode_wrapper(
        q.contiguous(), k.contiguous(), v.contiguous(), state.contiguous(),
        A_log.contiguous(), a.contiguous(), dt_bias.contiguous(), b.contiguous(),
        float(scale),
        output, new_state,
        int(batch_size), int(num_v_heads), int(num_q_heads), int(head_size)
    )

    return output.unsqueeze(1), new_state