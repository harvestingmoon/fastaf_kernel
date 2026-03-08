import os
import torch
from torch.utils.cpp_extension import load

try:
    from tvm_ffi import register_func
except ImportError:
    try:
        from tvm.ffi import register_func
    except ImportError:
        def register_func(_name):
            def deco(fn): return fn
            return deco

_cuda_module = None

def _load_cuda_module():
    global _cuda_module
    if _cuda_module is not None:
        return _cuda_module
    
    current_dir = os.path.dirname(__file__)
    wrapper_path = os.path.join(current_dir, 'wrapper.cpp')
    kernel_path = os.path.join(current_dir, 'kernel.cu')
    
    _cuda_module = load(
        name="gdn_kernel_optimized",
        sources=[wrapper_path, kernel_path],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math', '-Xptxas=-O3,-v'],
        verbose=False,
    )
    return _cuda_module

@register_func("kernel")
@register_func("flashinfer.kernel")
@torch.compiler.allow_in_graph
def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    module = _load_cuda_module()
    
    total_seq_len, q_heads, head_size = q.shape
    num_seqs = cu_seqlens.shape[0] - 1
    
    # Cast safety to prevent implicit graph breaks
    if a.dtype != torch.float32: a = a.to(torch.float32, copy=False)
    if b.dtype != torch.float32: b = b.to(torch.float32, copy=False)
    if dt_bias.dtype != torch.float32: dt_bias = dt_bias.to(torch.float32, copy=False)
    if A_log.dtype != torch.float32: A_log = A_log.to(torch.float32, copy=False)
    if cu_seqlens.dtype != torch.int32: cu_seqlens = cu_seqlens.to(torch.int32, copy=False)
    
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    if state is None:
        state = torch.empty(0, device=q.device, dtype=torch.float32)
    else:
        state = state.contiguous()

    # Note: We no longer duplicate K and V in Python for GQA. The CUDA kernel handles it natively!
    output = torch.empty((total_seq_len, q_heads, head_size), dtype=q.dtype, device=q.device)
    new_state = torch.empty((num_seqs, q_heads, head_size, head_size), dtype=torch.float32, device=q.device)
    
    module.kernel(
        q, k, v, state,
        A_log.contiguous(), a.contiguous(), dt_bias.contiguous(), b.contiguous(), 
        cu_seqlens.contiguous(), float(scale), 
        output, new_state
    )
    
    return output, new_state