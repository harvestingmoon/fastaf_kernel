"""
Ultra-Optimized GDN Prefill & Decode Kernels for Ampere/Hopper

Optimizations applied:
1. Warp-level parallelism for matvec reduction
2. Vectorized loads (VEC_K parameter)
3. Double buffering for memory latency hiding
4. Fast activation approximations (optional)
5. Fused operations with loop unrolling
6. Better register usage and occupancy
"""

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# ═══════════════════════════════════════════════════════════════════════════════
# Enhanced Autotune Configurations
# ═══════════════════════════════════════════════════════════════════════════════

def _get_prefill_configs(HEAD_SIZE: int):
    """Generate autotune configs for prefill kernel."""
    configs = []
    for block_v in [32, 64, 128]:
        for num_warps in [4, 8]:
            for num_stages in [2, 3, 4]:
                # Skip high register pressure combinations
                if block_v == 128 and num_stages > 3:
                    continue
                    
                # Add vectorization options
                for vec_k in [1, 2, 4]:
                    if HEAD_SIZE % vec_k != 0:
                        continue
                    if block_v * vec_k > 256:  # Register pressure limit
                        continue
                        
                    configs.append(
                        triton.Config(
                            {"BLOCK_V": block_v, "VEC_K": vec_k},
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
    return configs

def _get_decode_configs(HEAD_SIZE: int):
    """Generate autotune configs for decode kernel."""
    configs = []
    for block_v in [32, 64, 128]:
        for num_warps in [2, 4]:
            for vec_k in [1, 2, 4]:
                if HEAD_SIZE % vec_k != 0:
                    continue
                configs.append(
                    triton.Config(
                        {"BLOCK_V": block_v, "VEC_K": vec_k},
                        num_warps=num_warps,
                        num_stages=1,
                    )
                )
    return configs

# ═══════════════════════════════════════════════════════════════════════════════
# Fast Activation Functions (Optional)
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _softplus_fast(x: tl.tensor) -> tl.tensor:
    """Fast softplus approximation."""
    return tl.where(
        x > 8.0,
        x,
        tl.where(x < -8.0, tl.exp(x), tl.log(1.0 + tl.exp(x)))
    )

@triton.jit
def _sigmoid_fast(x: tl.tensor) -> tl.tensor:
    """Fast sigmoid approximation."""
    abs_x = tl.abs(x)
    return 0.5 + 0.5 * x / (1.0 + abs_x)

@triton.jit
def _exp_fast_neg(x: tl.tensor) -> tl.tensor:
    """Fast exp for negative arguments (g_t computation)."""
    # exp(x) ≈ 1 + x + x²/2 for x ≤ 0
    return 1.0 + x * (1.0 + x * 0.5)

# ═══════════════════════════════════════════════════════════════════════════════
# Optimized Prefill Kernel
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _gdn_prefill_kernel_optimized(
    # Pointers
    Q_ptr, K_ptr, V_ptr,
    A_log_ptr, A_ptr, DtBias_ptr, B_ptr,
    StateIn_ptr, StateOut_ptr, Y_ptr,
    CuSeqlens_ptr,
    # Strides
    stride_qt, stride_qh, stride_qk,
    stride_kt, stride_kh, stride_kk,
    stride_vt, stride_vh, stride_vv,
    stride_at, stride_ah,
    stride_bt, stride_bh,
    stride_sn, stride_sh, stride_sv, stride_sk,
    stride_yt, stride_yh, stride_yv,
    # Parameters
    GV_RATIO: tl.constexpr,
    scale: tl.constexpr,
    # Constants
    HAS_STATE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    VEC_K: tl.constexpr,
    USE_FAST_ACT: tl.constexpr,
):
    """Optimized GDN prefill kernel."""
    
    # Program IDs
    pid_vtile = tl.program_id(0)
    pid_seq = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    # Head mapping
    q_head_idx = pid_h // GV_RATIO
    v_head_idx = pid_h
    
    # Sequence bounds
    seq_start = tl.load(CuSeqlens_ptr + pid_seq).to(tl.int32)
    seq_end = tl.load(CuSeqlens_ptr + pid_seq + 1).to(tl.int32)
    seq_len = seq_end - seq_start
    
    # Offsets
    offs_v = pid_vtile * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_k = tl.arange(0, HEAD_SIZE // VEC_K) * VEC_K
    
    mask_v = offs_v < HEAD_SIZE
    mask_k = offs_k[:, None] + tl.arange(0, VEC_K)[None, :] < HEAD_SIZE
    
    # Precompute constants
    A_log_val = tl.load(A_log_ptr + pid_h).to(tl.float32)
    exp_A_log = tl.exp(A_log_val)
    dt_bias_val = tl.load(DtBias_ptr + pid_h).to(tl.float32)
    
    # Base pointers
    k_base = K_ptr + q_head_idx * stride_kh
    q_base = Q_ptr + q_head_idx * stride_qh
    v_base = V_ptr + v_head_idx * stride_vh
    a_base = A_ptr + pid_h * stride_ah
    b_base = B_ptr + pid_h * stride_bh
    y_base = Y_ptr + pid_h * stride_yh
    
    # State initialization
    state_base_in = StateIn_ptr + pid_seq * stride_sn + pid_h * stride_sh
    state_base_out = StateOut_ptr + pid_seq * stride_sn + pid_h * stride_sh
    
    S = tl.zeros([BLOCK_V, HEAD_SIZE], dtype=tl.float32)
    if HAS_STATE:
        # Vectorized state load
        for i in range(0, HEAD_SIZE, VEC_K):
            offs_k_vec = i + tl.arange(0, VEC_K)
            mask_k_vec = offs_k_vec < HEAD_SIZE
            s_ptrs = state_base_in + offs_v[:, None] * stride_sv + offs_k_vec[None, :] * stride_sk
            S_vec = tl.load(s_ptrs, mask=mask_v[:, None] & mask_k_vec[None, :], other=0.0)
            S = tl.where(mask_k_vec[None, :], S_vec, S)
    
    # Token loop with double buffering
    if seq_len > 0:
        # Prefetch first token
        k_next_ptrs = k_base + seq_start * stride_kt + offs_k[:, None] * stride_kk
        v_next_ptrs = v_base + seq_start * stride_vt + offs_v * stride_vv
        
        k_next = tl.load(k_next_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        v_next = tl.load(v_next_ptrs, mask=mask_v, other=0.0).to(tl.float32)
    
    for i in range(seq_len):
        t = seq_start + i
        
        # Double buffering
        k_t = k_next
        v_t = v_next
        
        # Prefetch next
        if i + 1 < seq_len:
            k_next_ptrs = k_base + (t + 1) * stride_kt + offs_k[:, None] * stride_kk
            v_next_ptrs = v_base + (t + 1) * stride_vt + offs_v * stride_vv
            k_next = tl.load(k_next_ptrs, mask=mask_k, other=0.0).to(tl.float32)
            v_next = tl.load(v_next_ptrs, mask=mask_v, other=0.0).to(tl.float32)
        
        # Gate computation
        a_val = tl.load(a_base + t * stride_at).to(tl.float32)
        b_val = tl.load(b_base + t * stride_bt).to(tl.float32)
        
        if USE_FAST_ACT:
            sp = _softplus_fast(a_val + dt_bias_val)
            g_t = _exp_fast_neg(-exp_A_log * sp)
            beta_t = _sigmoid_fast(b_val)
        else:
            sp = tl.log(1.0 + tl.exp(a_val + dt_bias_val))
            g_t = tl.exp(-exp_A_log * sp)
            beta_t = 1.0 / (1.0 + tl.exp(-b_val))
        
        # Matvec with loop unrolling
        raw_old_v = tl.zeros([BLOCK_V], dtype=tl.float32)
        
        # Process in chunks for better ILP
        chunk_size = VEC_K * 4
        for j in range(0, HEAD_SIZE, chunk_size):
            # Load k chunks
            k_chunks = []
            for chunk in range(0, min(chunk_size, HEAD_SIZE - j), VEC_K):
                k_ptr = k_base + t * stride_kt + (j + chunk) * stride_kk
                k_chunk = tl.load(k_ptr)
                k_chunks.append(k_chunk)
            
            # Fused multiply-add
            if len(k_chunks) > 0:
                k_stack = tl.stack(k_chunks)
                S_chunk = S[:, j:j+len(k_chunks)*VEC_K:VEC_K]
                raw_old_v += tl.sum(S_chunk * k_stack, axis=1)
        
        old_v = g_t * raw_old_v
        delta_v = beta_t * (v_t - old_v)
        
        # Update state
        for j in range(0, HEAD_SIZE, VEC_K):
            k_chunk = tl.load(k_base + t * stride_kt + j * stride_kk)
            S[:, j:j+VEC_K] = g_t * S[:, j:j+VEC_K] + delta_v[:, None] * k_chunk[None, :]
        
        # Output computation
        q_ptrs = q_base + t * stride_qt + offs_k[:, None] * stride_qk
        q_t = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        
        y_t = tl.zeros([BLOCK_V], dtype=tl.float32)
        for j in range(0, HEAD_SIZE, VEC_K * 2):
            if j + VEC_K * 2 <= HEAD_SIZE:
                q_chunk0 = tl.load(q_base + t * stride_qt + (j + 0*VEC_K) * stride_qk)
                q_chunk1 = tl.load(q_base + t * stride_qt + (j + 1*VEC_K) * stride_qk)
                y_t += tl.sum(S[:, j:j+VEC_K*2:VEC_K] * tl.stack([q_chunk0, q_chunk1]), axis=1)
        
        y_t = y_t * scale
        
        # Store output
        y_ptrs = y_base + t * stride_yt + offs_v * stride_yv
        tl.store(y_ptrs, y_t.to(tl.bfloat16), mask=mask_v)
    
    # Store state
    for i in range(0, HEAD_SIZE, VEC_K):
        offs_k_vec = i + tl.arange(0, VEC_K)
        mask_k_vec = offs_k_vec < HEAD_SIZE
        s_out_ptrs = state_base_out + offs_v[:, None] * stride_sv + offs_k_vec[None, :] * stride_sk
        tl.store(s_out_ptrs, S[:, i:i+VEC_K].to(tl.float32), 
                mask=mask_v[:, None] & mask_k_vec[None, :])
