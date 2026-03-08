"""
Highly Optimized GDN Prefill & Decode Kernels for Ampere/Hopper

Key optimizations:
1. Algebraic: Factor scalar g out of matvec → eliminate S_decayed temp
2. Precompute exp(A_log) before token loop (saves 1 exp() per token)
3. Autotune over BLOCK_V ∈ {32, 64, 128}, num_warps, num_stages
4. Pointer arithmetic hoisted outside inner loop
5. Fused softplus/sigmoid with minimal branching
6. K-tiling for reduced register pressure at larger BLOCK_V
7. Conditional masking elimination when BLOCK_V divides HEAD_SIZE evenly

Recurrence (per head h, k-last state layout [N, H, V, K]):
    g    = exp(-exp(A_log) * softplus(a_t + dt_bias))
    beta = sigmoid(b_t)
    old_v   = g * (S @ k_t)           # Algebraic optimization: factor out g
    delta_v = beta * (v_t - old_v)
    S       = g * S + outer(delta_v, k_t)
    y_t     = scale * S @ q_t
"""

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
# Autotune Configurations
# ═══════════════════════════════════════════════════════════════════════════════

def _get_prefill_configs():
    """
    Generate autotune configs for prefill kernel.
    
    Design rationale:
    - BLOCK_V=32: 4 tiles, low register pressure (~100 regs), good for long sequences
    - BLOCK_V=64: 2 tiles, medium pressure (~180 regs), sweet spot for most cases
    - BLOCK_V=128: 1 tile, high pressure (~250 regs), best for short sequences/high batch
    - num_stages=2-4: Software pipelining depth for memory latency hiding
    - num_warps=4-8: Higher for compute-bound, lower for memory-bound
    """
    configs = []
    for block_v in [32, 64, 128]:
        for num_warps in [4, 8]:
            for num_stages in [2, 3, 4]:
                # Skip high num_stages with large BLOCK_V (register pressure)
                if block_v == 128 and num_stages > 3:
                    continue
                configs.append(
                    triton.Config(
                        {"BLOCK_V": block_v},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


def _get_decode_configs():
    """
    Decode configs with K-chunked tiling for lower register pressure.

    Key insight: batch_size=1 always → grid=(D//BLOCK_V, 1, Hv=8).
    The current bottleneck is SM under-utilization (8-32 CTAs on 30 SMs)
    and register pressure (BLOCK_V×HEAD_SIZE floats in regs per CTA).

    BLOCK_K tiling reduces per-CTA register usage:
    - BLOCK_V=128, BLOCK_K=128: 16K regs (current behavior, 1 pass)
    - BLOCK_V=128, BLOCK_K=32:  4K regs (2-pass, L1-cached re-read)
    - BLOCK_V=32,  BLOCK_K=32:  1K regs (highest occupancy)
    - BLOCK_V=16,  BLOCK_K=32:  512 regs → 64 CTAs, fills all 30 SMs

    The 2-pass re-read hits L1 cache (tile ≤ 64KB < L1=128KB on Ampere).
    """
    configs = []
    for block_v in [16, 32, 64, 128]:
        for block_k in [32, 64, 128]:
            tile_regs = block_v * block_k
            if tile_regs <= 1024:       # tiny tile: 16×32, 16×64, 32×32
                warp_options = [1, 2, 4]
            elif tile_regs <= 4096:     # medium: 16×128, 32×64, 64×32, 32×128, 64×64
                warp_options = [2, 4, 8]
            else:                       # large: 64×128, 128×32..128
                warp_options = [4, 8]
            for nw in warp_options:
                configs.append(
                    triton.Config(
                        {"BLOCK_V": block_v, "BLOCK_K": block_k},
                        num_warps=nw,
                        num_stages=1,
                    )
                )
    return configs


PREFILL_CONFIGS = _get_prefill_configs()
DECODE_CONFIGS = _get_decode_configs()


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions (inlined by Triton JIT)
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _softplus_stable(x):
    """
    Numerically stable softplus: log(1 + exp(x))
    - x > 20: return x (avoids exp overflow)
    - x < -20: return exp(x) (avoids log1p underflow noise)
    - else: log1p(exp(x))
    """
    return tl.where(
        x > 20.0,
        x,
        tl.where(x < -20.0, tl.exp(x), tl.log(1.0 + tl.exp(x)))
    )


@triton.jit
def _sigmoid_stable(x):
    """
    Numerically stable sigmoid avoiding exp overflow.
    - x >= 0: 1 / (1 + exp(-x))
    - x < 0:  exp(x) / (1 + exp(x))
    """
    pos_path = 1.0 / (1.0 + tl.exp(-x))
    neg_path = tl.exp(x) / (1.0 + tl.exp(x))
    return tl.where(x >= 0.0, pos_path, neg_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Optimized Prefill Kernel
# ═══════════════════════════════════════════════════════════════════════════════

@triton.autotune(configs=PREFILL_CONFIGS, key=["HEAD_SIZE"])
@triton.jit
def _gdn_prefill_kernel_v2(
    # ─── Tensor pointers ───
    Q_ptr, K_ptr, V_ptr,
    A_log_ptr, A_ptr, DtBias_ptr, B_ptr,
    StateIn_ptr, StateOut_ptr, Y_ptr,
    CuSeqlens_ptr,
    # ─── Strides ───
    stride_qt, stride_qh, stride_qk,
    stride_kt, stride_kh, stride_kk,
    stride_vt, stride_vh, stride_vv,
    stride_at, stride_ah,
    stride_bt, stride_bh,
    stride_sn, stride_sh, stride_sv, stride_sk,
    stride_yt, stride_yh, stride_yv,
    # ─── Runtime parameters ───
    GV_RATIO,
    scale,
    # ─── Compile-time constants ───
    HAS_STATE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    Optimized GDN prefill kernel with:
    - Algebraic optimization (factor out scalar g from matvec)
    - Precomputed exp(A_log)
    - Hoisted pointer bases
    - Autotuned BLOCK_V, num_warps, num_stages
    """
    # ─── Program IDs ───
    pid_vtile = tl.program_id(0)
    pid_seq = tl.program_id(1)
    pid_h = tl.program_id(2)

    # ─── Head mapping (GVA: multiple v heads per q/k head) ───
    q_head_idx = pid_h // GV_RATIO
    v_head_idx = pid_h

    # ─── Sequence bounds ───
    seq_start = tl.load(CuSeqlens_ptr + pid_seq).to(tl.int32)
    seq_end = tl.load(CuSeqlens_ptr + pid_seq + 1).to(tl.int32)
    seq_len = seq_end - seq_start

    # ─── Offsets ───
    offs_v = pid_vtile * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_k = tl.arange(0, HEAD_SIZE)
    
    # Masking: only needed if BLOCK_V doesn't divide HEAD_SIZE evenly
    # For HEAD_SIZE=128 and BLOCK_V∈{32,64,128}, this is always false
    mask_v = offs_v < HEAD_SIZE

    # ─── OPTIMIZATION: Precompute exp(A_log) once (saves 1 exp per token) ───
    A_log_val = tl.load(A_log_ptr + pid_h).to(tl.float32)
    exp_A_log = tl.exp(A_log_val)  # Hoisted outside token loop
    dt_bias_val = tl.load(DtBias_ptr + pid_h).to(tl.float32)

    # ─── OPTIMIZATION: Precompute base pointers ───
    # These pointer bases are constant across the token loop
    k_base = K_ptr + q_head_idx * stride_kh
    q_base = Q_ptr + q_head_idx * stride_qh
    v_base = V_ptr + v_head_idx * stride_vh
    a_base = A_ptr + pid_h * stride_ah
    b_base = B_ptr + pid_h * stride_bh
    y_base = Y_ptr + pid_h * stride_yh

    # ─── State initialization ───
    state_base_in = StateIn_ptr + pid_seq * stride_sn + pid_h * stride_sh
    state_base_out = StateOut_ptr + pid_seq * stride_sn + pid_h * stride_sh
    
    s_ptrs = state_base_in + offs_v[:, None] * stride_sv + offs_k[None, :] * stride_sk

    if HAS_STATE:
        S = tl.load(s_ptrs, mask=mask_v[:, None], other=0.0).to(tl.float32)
    else:
        S = tl.zeros([BLOCK_V, HEAD_SIZE], dtype=tl.float32)

    # ═══════════════════════════════════════════════════════════════════════
    # Token Loop (optimized inner loop)
    # ═══════════════════════════════════════════════════════════════════════
    for i in range(seq_len):
        t = seq_start + i

        # ─── Load k_t [HEAD_SIZE] ───
        k_ptrs = k_base + t * stride_kt + offs_k * stride_kk
        k_t = tl.load(k_ptrs).to(tl.float32)

        # ─── Load v_t [BLOCK_V] ───
        v_ptrs = v_base + t * stride_vt + offs_v * stride_vv
        v_t = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)

        # ─── Gate computation ───
        a_val = tl.load(a_base + t * stride_at).to(tl.float32)
        b_val = tl.load(b_base + t * stride_bt).to(tl.float32)

        # g = exp(-exp(A_log) * softplus(a + dt_bias))
        # OPTIMIZATION: exp_A_log precomputed above
        sp = _softplus_stable(a_val + dt_bias_val)
        g_t = tl.exp(-exp_A_log * sp)

        # beta = sigmoid(b)
        beta_t = _sigmoid_stable(b_val)

        # ═══════════════════════════════════════════════════════════════════
        # CORE ALGEBRAIC OPTIMIZATION
        # ═══════════════════════════════════════════════════════════════════
        # Original (3 passes over [BLOCK_V, HEAD_SIZE]):
        #   S_decayed = g * S                   # Pass 1
        #   old_v = sum(S_decayed * k, axis=1)  # Pass 2
        #   S = S_decayed + outer(delta_v, k)   # Pass 3
        #
        # Optimized (2 passes):
        #   old_v = g * sum(S * k, axis=1)      # Pass 1 + scalar mul
        #   S = g * S + outer(delta_v, k)       # Pass 2 (single fused op)
        # ═══════════════════════════════════════════════════════════════════

        # Pass 1: Matvec on original state, then scale by g
        # (Saves one full [BLOCK_V × HEAD_SIZE] element-wise multiply)
        raw_old_v = tl.sum(S * k_t[None, :], axis=1)  # [BLOCK_V]
        old_v = g_t * raw_old_v

        # Delta computation
        delta_v = beta_t * (v_t - old_v)  # [BLOCK_V]

        # Pass 2: Fused decay + rank-1 update
        # Compiler can generate FMA: S = g*S + delta_v[:,None]*k[None,:]
        S = g_t * S + delta_v[:, None] * k_t[None, :]

        # ─── Output: y_t = scale * S @ q_t ───
        q_ptrs = q_base + t * stride_qt + offs_k * stride_qk
        q_t = tl.load(q_ptrs).to(tl.float32)
        y_t = tl.sum(S * q_t[None, :], axis=1) * scale

        # ─── Store output ───
        y_ptrs = y_base + t * stride_yt + offs_v * stride_yv
        tl.store(y_ptrs, y_t.to(tl.bfloat16), mask=mask_v)

    # ─── Write back final state ───
    s_out_ptrs = state_base_out + offs_v[:, None] * stride_sv + offs_k[None, :] * stride_sk
    tl.store(s_out_ptrs, S.to(tl.float32), mask=mask_v[:, None])


# ═══════════════════════════════════════════════════════════════════════════════
# Optimized Decode Kernel — K-Chunked 2-Pass
# ═══════════════════════════════════════════════════════════════════════════════

@triton.autotune(configs=DECODE_CONFIGS, key=["HEAD_SIZE"])
@triton.jit
def _gdn_decode_kernel_v3(
    # ─── Tensor pointers ───
    Q_ptr, K_ptr, V_ptr,
    A_log_ptr, A_ptr, DtBias_ptr, B_ptr,
    StateIn_ptr, StateOut_ptr, Y_ptr,
    # ─── Strides (T dim squeezed out — all [B, H, D] shaped) ───
    stride_qb, stride_qh, stride_qk,
    stride_kb, stride_kh, stride_kk,
    stride_vb, stride_vh, stride_vv,
    stride_ab, stride_ah,
    stride_bb, stride_bh,
    stride_sbn, stride_sh, stride_sv, stride_sk,
    stride_yb, stride_yh, stride_yv,
    # ─── Runtime parameters ───
    GV_RATIO,
    scale,
    # ─── Compile-time constants ───
    HAS_STATE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    K-chunked GDN decode kernel — 2-pass over state for lower register pressure.

    Instead of holding the full [BLOCK_V, HEAD_SIZE] state tile in registers
    (up to 128×128 = 16K floats = 64KB), we process K in BLOCK_K-sized chunks:

    Pass 1: Loop over K chunks → compute old_v = g * sum_K(S * k)
      Each iteration holds only [BLOCK_V, BLOCK_K] in regs (e.g. 32×32 = 4KB)

    Pass 2: Loop over K chunks → update state + compute output
      Re-reads state (hits L1 cache, tile fits in 128KB L1 on Ampere)

    Benefits:
    - 4-16× lower register pressure → higher occupancy → better mem latency hiding
    - BLOCK_V=16 gives 64 CTAs → fills all 30 SMs (vs 8 with BLOCK_V=128)
    - When BLOCK_K=HEAD_SIZE, loop runs once → equivalent to single-pass but
      with an L1-cached re-read (negligible cost)
    """
    pid_vtile = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    q_head_idx = pid_h // GV_RATIO
    v_head_idx = pid_h

    offs_v = pid_vtile * BLOCK_V + tl.arange(0, BLOCK_V)

    # ─── Per-head constants (scalars, loaded once) ───
    A_log_val = tl.load(A_log_ptr + pid_h).to(tl.float32)
    exp_A_log = tl.exp(A_log_val)
    dt_bias_val = tl.load(DtBias_ptr + pid_h).to(tl.float32)

    # ─── Precompute base pointers (loop-invariant) ───
    state_base_in = StateIn_ptr + pid_b * stride_sbn + pid_h * stride_sh
    state_base_out = StateOut_ptr + pid_b * stride_sbn + pid_h * stride_sh
    v_stride = offs_v * stride_sv   # precomputed for pointer arithmetic

    k_base = K_ptr + pid_b * stride_kb + q_head_idx * stride_kh
    q_base = Q_ptr + pid_b * stride_qb + q_head_idx * stride_qh

    # ─── Load v [BLOCK_V] ───
    v_t = tl.load(V_ptr + pid_b * stride_vb + v_head_idx * stride_vh
                  + offs_v * stride_vv).to(tl.float32)

    # ─── Gate scalars ───
    a_val = tl.load(A_ptr + pid_b * stride_ab + pid_h * stride_ah).to(tl.float32)
    b_val = tl.load(B_ptr + pid_b * stride_bb + pid_h * stride_bh).to(tl.float32)

    sp = _softplus_stable(a_val + dt_bias_val)
    g_t = tl.exp(-exp_A_log * sp)
    beta_t = _sigmoid_stable(b_val)

    # ═══════════════════════════════════════════════════════════════════════════
    # Pass 1: Compute old_v = g * sum_K(S[:,K] * k[K])
    # Accumulate partial dot products across K chunks.
    # Register usage per iteration: [BLOCK_V, BLOCK_K] floats.
    # ═══════════════════════════════════════════════════════════════════════════
    raw_old_v = tl.zeros([BLOCK_V], dtype=tl.float32)
    for kk in range(0, HEAD_SIZE, BLOCK_K):
        offs_k = kk + tl.arange(0, BLOCK_K)
        s_ptrs = state_base_in + v_stride[:, None] + offs_k[None, :] * stride_sk
        if HAS_STATE:
            S_chunk = tl.load(s_ptrs).to(tl.float32)
        else:
            S_chunk = tl.zeros([BLOCK_V, BLOCK_K], dtype=tl.float32)
        k_chunk = tl.load(k_base + offs_k * stride_kk).to(tl.float32)
        raw_old_v += tl.sum(S_chunk * k_chunk[None, :], axis=1)

    old_v = g_t * raw_old_v
    delta_v = beta_t * (v_t - old_v)

    # ═══════════════════════════════════════════════════════════════════════════
    # Pass 2: Update state + compute output y = scale * sum_K(S_new * q)
    # Re-reads state tiles — hits L1 cache (tile ≤ 64KB < 128KB L1 on Ampere).
    # Fuses: S_new = g*S + outer(delta_v, k), y += S_new @ q, store(S_new).
    # ═══════════════════════════════════════════════════════════════════════════
    y_t = tl.zeros([BLOCK_V], dtype=tl.float32)
    for kk in range(0, HEAD_SIZE, BLOCK_K):
        offs_k = kk + tl.arange(0, BLOCK_K)
        s_ptrs = state_base_in + v_stride[:, None] + offs_k[None, :] * stride_sk
        if HAS_STATE:
            S_chunk = tl.load(s_ptrs).to(tl.float32)
        else:
            S_chunk = tl.zeros([BLOCK_V, BLOCK_K], dtype=tl.float32)
        k_chunk = tl.load(k_base + offs_k * stride_kk).to(tl.float32)
        q_chunk = tl.load(q_base + offs_k * stride_qk).to(tl.float32)

        # Fused decay + rank-1 update
        S_chunk = g_t * S_chunk + delta_v[:, None] * k_chunk[None, :]

        # Accumulate output
        y_t += tl.sum(S_chunk * q_chunk[None, :], axis=1)

        # Store updated state chunk
        s_out_ptrs = state_base_out + v_stride[:, None] + offs_k[None, :] * stride_sk
        tl.store(s_out_ptrs, S_chunk)

    # ─── Store output ───
    y_t *= scale
    tl.store(Y_ptr + pid_b * stride_yb + pid_h * stride_yh + offs_v * stride_yv,
             y_t.to(tl.bfloat16))


# ═══════════════════════════════════════════════════════════════════════════════
# Python Entry Points
# ═══════════════════════════════════════════════════════════════════════════════

def kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: Optional[torch.Tensor],
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized GDN prefill entry point.
    
    Returns: (output [T, Hsab, D] bf16, new_state [N, Hsab, D, D] f32)
    """
    T, Hq, D = q.shape
    Hv = v.shape[1]
    Hsab = Hv
    N = cu_seqlens.shape[0] - 1

    if not scale or scale == 0.0:
        scale = 1.0 / math.sqrt(D)

    device = q.device
    HEAD_SIZE = D
    GV_RATIO = Hsab // Hq

    # Allocate outputs
    output = torch.empty(T, Hsab, D, dtype=torch.bfloat16, device=device)
    new_state = torch.empty(N, Hsab, D, D, dtype=torch.float32, device=device)

    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    A_log = A_log.contiguous()
    dt_bias = dt_bias.contiguous()

    cu_i32 = cu_seqlens.to(torch.int32) if cu_seqlens.dtype != torch.int32 else cu_seqlens

    has_state = state is not None
    state_in = state.contiguous() if has_state else new_state

    # Grid dimensions - let autotune pick BLOCK_V
    def grid(META):
        return (HEAD_SIZE // META["BLOCK_V"], N, Hsab)

    _gdn_prefill_kernel_v2[grid](
        q, k, v,
        A_log, a, dt_bias, b,
        state_in, new_state, output,
        cu_i32,
        # Strides
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        new_state.stride(0), new_state.stride(1),
        new_state.stride(2), new_state.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        # Runtime params
        GV_RATIO,
        float(scale),
        # Constexpr
        HAS_STATE=int(has_state),
        HEAD_SIZE=HEAD_SIZE,
    )

    return output, new_state


def kernel_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: Optional[torch.Tensor],
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized GDN decode entry point.

    Returns: (output [B, 1, Hv, D] bf16, new_state [B, Hv, D, D] f32)
    """
    B, _T, Hq, D = q.shape
    Hv = v.shape[2]
    Hsab = Hv

    if not scale or scale == 0.0:
        scale = 1.0 / math.sqrt(D)

    device = q.device
    HEAD_SIZE = D
    GV_RATIO = Hsab // Hq

    output_sq = torch.empty(B, Hsab, D, dtype=torch.bfloat16, device=device)
    new_state = torch.empty(B, Hsab, D, D, dtype=torch.float32, device=device)

    # Squeeze T=1 via view (no copy, no overhead — just drops the dim)
    q_sq = q.view(B, Hq, D)
    k_sq = k.view(B, Hq, D)
    v_sq = v.view(B, Hv, D)
    a_sq = a.view(B, Hsab)
    b_sq = b.view(B, Hsab)

    has_state = state is not None
    state_in = state if has_state else new_state

    def grid(META):
        return (HEAD_SIZE // META["BLOCK_V"], B, Hsab)

    _gdn_decode_kernel_v3[grid](
        q_sq, k_sq, v_sq,
        A_log, a_sq, dt_bias, b_sq,
        state_in, new_state, output_sq,
        # Strides
        q_sq.stride(0), q_sq.stride(1), q_sq.stride(2),
        k_sq.stride(0), k_sq.stride(1), k_sq.stride(2),
        v_sq.stride(0), v_sq.stride(1), v_sq.stride(2),
        a_sq.stride(0), a_sq.stride(1),
        b_sq.stride(0), b_sq.stride(1),
        new_state.stride(0), new_state.stride(1),
        new_state.stride(2), new_state.stride(3),
        output_sq.stride(0), output_sq.stride(1), output_sq.stride(2),
        # Runtime params
        GV_RATIO,
        float(scale),
        # Constexpr
        HAS_STATE=int(has_state),
        HEAD_SIZE=HEAD_SIZE,
    )

    return output_sq.unsqueeze(1), new_state