"""
GDN (Gated Delta Rule) Prefill kernel for Ampere (SM80/SM86).

Uses Triton for GPU acceleration. This is the benchmark entry point.

Recurrence (per head h, k-last state layout [N, H, V, K]):
    g    = exp(-exp(A_log) * softplus(a_t + dt_bias))   scalar per (t, h)
    beta = sigmoid(b_t)                                   scalar per (t, h)

    State S in [V, K] layout (k-last: V rows, K columns):
        old_v   = S @ k_t              [V]    row-wise dot with k
        delta_v = beta * (v_t - old_v) [V]
        S_new   = g * S + outer(delta_v, k_t)  [V, K]
        y_t     = scale * S_new @ q_t  [V]

Kernel design:
    Grid: (HEAD_SIZE // BLOCK_V, N, Hsab)
     - pid0: V-tile index
     - pid1: sequence index
     - pid2: sab_head index
    Each program handles one V-tile of [BLOCK_V, K=128] state for one (seq, head).
    State tile kept in registers across the token loop → no global state r/w per token.
    Sequences run to variable length via cu_seqlens (int32 in kernel, int64 outside).

Input signature (benchmark framework):
    kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)
    q:         [T, Hq=4, D=128]          bf16
    k:         [T, Hk=4, D=128]          bf16
    v:         [T, Hv=8, D=128]          bf16
    state:     [N, Hv=8, D=128, D=128]   f32   k-last [N,H,V,K]  (or None)
    A_log:     [Hv=8]                    f32
    a:         [T, Hv=8]                 bf16
    dt_bias:   [Hv=8]                    f32
    b:         [T, Hv=8]                 bf16
    cu_seqlens:[N+1]                     int64
    scale:     float scalar

Returns: (output [T, Hv=8, D=128] bf16,  new_state [N, Hv=8, D=128, D=128] f32)
"""

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: one program per (v_tile, seq, head)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _gdn_prefill_kernel(
    # ---- Global tensor pointers ----
    Q_ptr,          # [T, Hq, K]   bf16
    K_ptr,          # [T, Hk, K]   bf16
    V_ptr,          # [T, Hv, V]   bf16
    A_log_ptr,      # [Hsab]        f32
    A_ptr,          # [T, Hsab]     bf16
    DtBias_ptr,     # [Hsab]        f32
    B_ptr,          # [T, Hsab]     bf16
    StateIn_ptr,    # [N, Hsab, V, K]  f32
    StateOut_ptr,   # [N, Hsab, V, K]  f32
    Y_ptr,          # [T, Hsab, V]  bf16
    CuSeqlens_ptr,  # [N+1]  int32
    # ---- Strides ----
    stride_qt, stride_qh, stride_qk,
    stride_kt, stride_kh, stride_kk,
    stride_vt, stride_vh, stride_vv,
    stride_at, stride_ah,
    stride_bt, stride_bh,
    stride_sn, stride_sh, stride_sv, stride_sk,
    stride_yt, stride_yh, stride_yv,
    # ---- Runtime dims ----
    GV_RATIO,       # Hsab // Hq  (=2 for Hq=4, Hv=8)
    scale,
    HAS_STATE: tl.constexpr,   # 1 if initial state provided, else 0 (init to zero)
    HEAD_SIZE: tl.constexpr,   # K = V = D = 128
    BLOCK_V: tl.constexpr,     # rows per program (e.g. 32)
):
    pid_vtile = tl.program_id(0)   # which V-row tile
    pid_seq   = tl.program_id(1)   # sequence index
    pid_h     = tl.program_id(2)   # sab_head index

    # GVA head mapping
    q_head_idx = pid_h // GV_RATIO   # q/k head
    v_head_idx = pid_h                # v head (v_head == sab_head since Hv==Hsab)

    # Sequence token range
    seq_start = tl.load(CuSeqlens_ptr + pid_seq).to(tl.int32)
    seq_end   = tl.load(CuSeqlens_ptr + pid_seq + 1).to(tl.int32)
    seq_len   = seq_end - seq_start

    # Offsets
    offs_v = pid_vtile * BLOCK_V + tl.arange(0, BLOCK_V)   # [BLOCK_V]
    mask_v = offs_v < HEAD_SIZE
    offs_k = tl.arange(0, HEAD_SIZE)                        # [HEAD_SIZE]

    # Per-head learnable scalars
    A_log_val = tl.load(A_log_ptr + pid_h).to(tl.float32)
    dt_bias_val = tl.load(DtBias_ptr + pid_h).to(tl.float32)

    # Load (or zero-init) state tile S[offs_v, offs_k] — shape [BLOCK_V, HEAD_SIZE]
    state_base_in  = StateIn_ptr  + pid_seq * stride_sn + pid_h * stride_sh
    state_base_out = StateOut_ptr + pid_seq * stride_sn + pid_h * stride_sh
    s_ptrs = state_base_in + offs_v[:, None] * stride_sv + offs_k[None, :] * stride_sk

    if HAS_STATE:
        S = tl.load(s_ptrs, mask=mask_v[:, None], other=0.0)
    else:
        S = tl.zeros([BLOCK_V, HEAD_SIZE], dtype=tl.float32)

    # ── Token loop ──────────────────────────────────────────────────────────
    for i in range(seq_len):
        t = seq_start + i

        # Load k_t [HEAD_SIZE] (q/k share the same head index in GVA)
        k_ptrs = K_ptr + t * stride_kt + q_head_idx * stride_kh + offs_k * stride_kk
        k_t = tl.load(k_ptrs).to(tl.float32)

        # Load v_t [BLOCK_V] (full V slice for this head's tile)
        v_ptrs = V_ptr + t * stride_vt + v_head_idx * stride_vh + offs_v * stride_vv
        v_t = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)

        # Gate computation: g = exp(-exp(A_log) * softplus(a + dt_bias))
        #                   beta = sigmoid(b)
        a_val = tl.load(A_ptr + t * stride_at + pid_h * stride_ah).to(tl.float32)
        b_val = tl.load(B_ptr + t * stride_bt + pid_h * stride_bh).to(tl.float32)
        x = a_val + dt_bias_val
        # Match CUDA/reference numerics closely:
        # softplus(x): x>20 -> x, x<-20 -> exp(x), else log1p(exp(x))
        sp_mid = tl.log(1.0 + tl.exp(x))
        sp = tl.where(x > 20.0, x, tl.where(x < -20.0, tl.exp(x), sp_mid))
        g_t = tl.exp(-tl.exp(A_log_val) * sp)

        # Stable sigmoid with explicit branches (matches CUDA implementation)
        z_pos = tl.exp(-b_val)
        sig_pos = 1.0 / (1.0 + z_pos)
        z_neg = tl.exp(b_val)
        sig_neg = z_neg / (1.0 + z_neg)
        beta_t = tl.where(b_val >= 0.0, sig_pos, sig_neg)

        # Reference ordering:
        #   old_state = g * state
        #   old_v     = old_state @ k
        #   state_new = old_state + outer(beta * (v - old_v), k)
        S_decayed = g_t * S

        # old_v = (g * S) @ k   [BLOCK_V]
        old_v = tl.sum(S_decayed * k_t[None, :], axis=1)

        # delta_v = beta * (v_t - old_v)  [BLOCK_V]
        delta_v = beta_t * (v_t - old_v)

        # S = old_state + outer(delta_v, k_t)  [BLOCK_V, HEAD_SIZE]
        S = S_decayed + delta_v[:, None] * k_t[None, :]

        # y_t = scale * S @ q_t   [BLOCK_V]
        q_ptrs = Q_ptr + t * stride_qt + q_head_idx * stride_qh + offs_k * stride_qk
        q_t = tl.load(q_ptrs).to(tl.float32)
        y_t = tl.sum(S * q_t[None, :], axis=1) * scale

        # Store y_t -> Y[t, pid_h, offs_v]
        y_ptrs = Y_ptr + t * stride_yt + pid_h * stride_yh + offs_v * stride_yv
        tl.store(y_ptrs, y_t.to(tl.bfloat16), mask=mask_v)

    # ── Write back final state tile ─────────────────────────────────────────
    s_out_ptrs = state_base_out + offs_v[:, None] * stride_sv + offs_k[None, :] * stride_sk
    tl.store(s_out_ptrs, S, mask=mask_v[:, None])


# ─────────────────────────────────────────────────────────────────────────────
# Python entry point — matches benchmark framework signature exactly
# ─────────────────────────────────────────────────────────────────────────────

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
    GDN prefill entry point called by the benchmark framework.

    Args match the definition inputs: q,k,v bf16; state f32 k-last or None;
    A_log,dt_bias f32; a,b bf16; cu_seqlens int64; scale float.

    Returns: (output [T, Hsab, D] bf16,  new_state [N, Hsab, D, D] f32)
    """
    T, Hq, D = q.shape
    Hv = v.shape[1]
    Hsab = Hv            # GVA: num_sab_heads = Hv (= max(Hq, Hv))
    N = int(cu_seqlens.shape[0]) - 1

    if not scale or scale == 0.0:
        scale = 1.0 / math.sqrt(D)

    device = q.device
    HEAD_SIZE = D    # 128
    BLOCK_V   = 32   # 4 tiles × 32 = 128; tune to 64 if register pressure allows

    GV_RATIO = Hsab // Hq   # = 2

    # Allocate outputs
    output    = torch.empty(T, Hsab, D, dtype=torch.bfloat16, device=device)
    new_state = torch.empty(N, Hsab, D, D, dtype=torch.float32, device=device)

    # Contiguous inputs (cheap no-op if already contiguous)
    q        = q.contiguous()
    k        = k.contiguous()
    v        = v.contiguous()
    a        = a.contiguous()
    b        = b.contiguous()
    A_log    = A_log.contiguous()
    dt_bias  = dt_bias.contiguous()

    # cu_seqlens as int32 for the kernel (int64 → int32 cast)
    cu_i32 = cu_seqlens.to(torch.int32) if cu_seqlens.dtype != torch.int32 else cu_seqlens

    has_state = (state is not None)
    if has_state:
        state_in = state.contiguous()
    else:
        # Dummy pointer — won't be touched (HAS_STATE=False path)
        state_in = new_state

    num_v_tiles = HEAD_SIZE // BLOCK_V   # = 4
    grid = (num_v_tiles, N, Hsab)

    _gdn_prefill_kernel[grid](
        q, k, v,
        A_log, a, dt_bias, b,
        state_in,
        new_state,
        output,
        cu_i32,
        # Q strides [T, Hq, D]
        q.stride(0), q.stride(1), q.stride(2),
        # K strides [T, Hk, D]
        k.stride(0), k.stride(1), k.stride(2),
        # V strides [T, Hv, D]
        v.stride(0), v.stride(1), v.stride(2),
        # A strides [T, Hsab]
        a.stride(0), a.stride(1),
        # B strides [T, Hsab]
        b.stride(0), b.stride(1),
        # State strides [N, Hsab, V, K]
        new_state.stride(0), new_state.stride(1),
        new_state.stride(2), new_state.stride(3),
        # Y strides [T, Hsab, D]
        output.stride(0), output.stride(1), output.stride(2),
        GV_RATIO,
        float(scale),
        HAS_STATE=int(has_state),
        HEAD_SIZE=HEAD_SIZE,
        BLOCK_V=BLOCK_V,
    )

    return output, new_state


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: GDN decode — one program per (v_tile, batch_item, head)
# T=1 always; no cu_seqlens; q/k/v have shape [B, 1, H, D]
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _gdn_decode_kernel(
    # ---- Global tensor pointers ----
    Q_ptr,          # [B, 1, Hq, K]    bf16
    K_ptr,          # [B, 1, Hk, K]    bf16
    V_ptr,          # [B, 1, Hv, V]    bf16
    A_log_ptr,      # [Hsab]            f32
    A_ptr,          # [B, 1, Hsab]      bf16
    DtBias_ptr,     # [Hsab]            f32
    B_ptr,          # [B, 1, Hsab]      bf16
    StateIn_ptr,    # [B, Hsab, V, K]   f32
    StateOut_ptr,   # [B, Hsab, V, K]   f32
    Y_ptr,          # [B, 1, Hsab, V]   bf16
    # ---- Strides ----
    stride_qb, stride_qt, stride_qh, stride_qk,
    stride_kb, stride_kt, stride_kh, stride_kk,
    stride_vb, stride_vt, stride_vh, stride_vv,
    stride_ab, stride_at, stride_ah,
    stride_bb, stride_bt, stride_bh,
    stride_sbn, stride_sh, stride_sv, stride_sk,
    stride_yb, stride_yt, stride_yh, stride_yv,
    # ---- Runtime dims ----
    GV_RATIO,
    scale,
    HAS_STATE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_vtile = tl.program_id(0)   # V-row tile index
    pid_b     = tl.program_id(1)   # batch index
    pid_h     = tl.program_id(2)   # sab_head index

    # GVA head mapping
    q_head_idx = pid_h // GV_RATIO
    v_head_idx = pid_h

    offs_v = pid_vtile * BLOCK_V + tl.arange(0, BLOCK_V)   # [BLOCK_V]
    mask_v = offs_v < HEAD_SIZE
    offs_k = tl.arange(0, HEAD_SIZE)                        # [HEAD_SIZE]

    # Per-head learnable scalars
    A_log_val   = tl.load(A_log_ptr + pid_h).to(tl.float32)
    dt_bias_val = tl.load(DtBias_ptr + pid_h).to(tl.float32)

    # Load (or zero-init) state tile S[offs_v, offs_k] — [BLOCK_V, HEAD_SIZE]
    state_base_in  = StateIn_ptr  + pid_b * stride_sbn + pid_h * stride_sh
    state_base_out = StateOut_ptr + pid_b * stride_sbn + pid_h * stride_sh
    s_ptrs = state_base_in + offs_v[:, None] * stride_sv + offs_k[None, :] * stride_sk

    if HAS_STATE:
        S = tl.load(s_ptrs, mask=mask_v[:, None], other=0.0)
    else:
        S = tl.zeros([BLOCK_V, HEAD_SIZE], dtype=tl.float32)

    # Load k [HEAD_SIZE] — single token (seq offset = 0)
    k_ptrs = K_ptr + pid_b * stride_kb + 0 * stride_kt + q_head_idx * stride_kh + offs_k * stride_kk
    k_t = tl.load(k_ptrs).to(tl.float32)

    # Load v [BLOCK_V]
    v_ptrs = V_ptr + pid_b * stride_vb + 0 * stride_vt + v_head_idx * stride_vh + offs_v * stride_vv
    v_t = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)

    # Gate scalars: g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)
    a_val = tl.load(A_ptr + pid_b * stride_ab + 0 * stride_at + pid_h * stride_ah).to(tl.float32)
    b_val = tl.load(B_ptr + pid_b * stride_bb + 0 * stride_bt + pid_h * stride_bh).to(tl.float32)
    x = a_val + dt_bias_val
    sp_mid = tl.log(1.0 + tl.exp(x))
    sp = tl.where(x > 20.0, x, tl.where(x < -20.0, tl.exp(x), sp_mid))
    g_t = tl.exp(-tl.exp(A_log_val) * sp)

    z_pos = tl.exp(-b_val)
    sig_pos = 1.0 / (1.0 + z_pos)
    z_neg = tl.exp(b_val)
    sig_neg = z_neg / (1.0 + z_neg)
    beta_t = tl.where(b_val >= 0.0, sig_pos, sig_neg)

    # Delta rule (same as prefill body, VK layout, k-last):
    #   S_decayed = g * S
    #   old_v     = S_decayed @ k    [BLOCK_V]
    #   delta_v   = beta * (v - old_v)
    #   S_new     = S_decayed + outer(delta_v, k)
    #   y         = scale * S_new @ q
    S_decayed = g_t * S
    old_v   = tl.sum(S_decayed * k_t[None, :], axis=1)
    delta_v = beta_t * (v_t - old_v)
    S = S_decayed + delta_v[:, None] * k_t[None, :]

    q_ptrs = Q_ptr + pid_b * stride_qb + 0 * stride_qt + q_head_idx * stride_qh + offs_k * stride_qk
    q_t = tl.load(q_ptrs).to(tl.float32)
    y_t = tl.sum(S * q_t[None, :], axis=1) * scale

    # Store y_t -> Y[pid_b, 0, pid_h, offs_v]
    y_ptrs = Y_ptr + pid_b * stride_yb + 0 * stride_yt + pid_h * stride_yh + offs_v * stride_yv
    tl.store(y_ptrs, y_t.to(tl.bfloat16), mask=mask_v)

    # Write back updated state tile
    s_out_ptrs = state_base_out + offs_v[:, None] * stride_sv + offs_k[None, :] * stride_sk
    tl.store(s_out_ptrs, S, mask=mask_v[:, None])


# ─────────────────────────────────────────────────────────────────────────────
# Python decode entry point — matches benchmark framework signature exactly
# ─────────────────────────────────────────────────────────────────────────────

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
    GDN decode entry point called by the benchmark framework.

    Args: q,k,v bf16 [B,1,H,D]; state f32 [B,Hv,D,D] k-last or None;
    A_log,dt_bias f32; a,b bf16 [B,1,Hv]; scale float.

    Returns: (output [B, 1, Hv, D] bf16,  new_state [B, Hv, D, D] f32)
    """
    B, _T, Hq, D = q.shape   # _T == 1 always for decode
    Hv   = v.shape[2]
    Hsab = Hv

    if not scale or scale == 0.0:
        scale = 1.0 / math.sqrt(D)

    device    = q.device
    HEAD_SIZE = D      # 128
    BLOCK_V   = 32
    GV_RATIO  = Hsab // Hq   # = 2

    # Allocate outputs
    output    = torch.empty(B, 1, Hsab, D, dtype=torch.bfloat16, device=device)
    new_state = torch.empty(B, Hsab, D, D, dtype=torch.float32, device=device)

    q       = q.contiguous()
    k       = k.contiguous()
    v       = v.contiguous()
    a       = a.contiguous()
    b       = b.contiguous()
    A_log   = A_log.contiguous()
    dt_bias = dt_bias.contiguous()

    has_state = (state is not None)
    state_in  = state.contiguous() if has_state else new_state

    num_v_tiles = HEAD_SIZE // BLOCK_V   # = 4
    grid = (num_v_tiles, B, Hsab)

    _gdn_decode_kernel[grid](
        q, k, v,
        A_log, a, dt_bias, b,
        state_in,
        new_state,
        output,
        # Q strides [B, 1, Hq, D]
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # K strides [B, 1, Hk, D]
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # V strides [B, 1, Hv, D]
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # A strides [B, 1, Hsab]
        a.stride(0), a.stride(1), a.stride(2),
        # B strides [B, 1, Hsab]
        b.stride(0), b.stride(1), b.stride(2),
        # State strides [B, Hsab, V, K]
        new_state.stride(0), new_state.stride(1),
        new_state.stride(2), new_state.stride(3),
        # Y strides [B, 1, Hsab, D]
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        GV_RATIO,
        float(scale),
        HAS_STATE=int(has_state),
        HEAD_SIZE=HEAD_SIZE,
        BLOCK_V=BLOCK_V,
    )

    return output, new_state


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test / correctness check against reference
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch.nn.functional as F

    torch.manual_seed(42)
    device = "cuda"

    # Match benchmark config: Hq=4, Hv=8, D=128
    Hq, Hv, D = 4, 8, 128
    T, N = 64, 2
    cu_seqlens = torch.tensor([0, 32, 64], dtype=torch.int64, device=device)

    q     = torch.randn(T, Hq, D, device=device, dtype=torch.bfloat16) * 0.1
    k     = torch.randn(T, Hq, D, device=device, dtype=torch.bfloat16) * 0.1
    v     = torch.randn(T, Hv, D, device=device, dtype=torch.bfloat16) * 0.1
    # Use stable gates: A_log=-1 → exp(A_log)≈0.37; dt_bias=1 keeps g<1
    A_log   = torch.full((Hv,), -1.0, device=device, dtype=torch.float32)
    a_in  = torch.randn(T, Hv, device=device, dtype=torch.bfloat16) * 0.3
    dt_bias = torch.ones(Hv, device=device, dtype=torch.float32)
    b_in  = torch.randn(T, Hv, device=device, dtype=torch.bfloat16) * 0.3
    state = torch.zeros(N, Hv, D, D, device=device, dtype=torch.float32)
    scale_val = 1.0 / math.sqrt(D)

    print(f"Testing on {torch.cuda.get_device_name()}")
    out, ns = kernel(q, k, v, state, A_log, a_in, dt_bias, b_in, cu_seqlens, scale_val)
    print(f"[prefill] output:    {out.shape}  dtype={out.dtype}")
    print(f"[prefill] new_state: {ns.shape}   dtype={ns.dtype}")
    print(f"[prefill] output stats: mean={out.float().mean():.4f}, std={out.float().std():.4f}")

    # ── Decode smoke test ────────────────────────────────────────────────────
    print("\n--- Decode smoke test ---")
    B = 4  # decode: batch_size sequences, each with exactly 1 token
    q_d   = torch.randn(B, 1, Hq, D, device=device, dtype=torch.bfloat16) * 0.1
    k_d   = torch.randn(B, 1, Hq, D, device=device, dtype=torch.bfloat16) * 0.1
    v_d   = torch.randn(B, 1, Hv, D, device=device, dtype=torch.bfloat16) * 0.1
    a_d   = torch.randn(B, 1, Hv, device=device, dtype=torch.bfloat16) * 0.3
    b_d   = torch.randn(B, 1, Hv, device=device, dtype=torch.bfloat16) * 0.3
    state_d = torch.zeros(B, Hv, D, D, device=device, dtype=torch.float32)

    out_d, ns_d = kernel_decode(q_d, k_d, v_d, state_d, A_log, a_d, dt_bias, b_d, scale_val)
    print(f"[decode] output:    {out_d.shape}  dtype={out_d.dtype}")
    print(f"[decode] new_state: {ns_d.shape}   dtype={ns_d.dtype}")
    print(f"[decode] output stats: mean={out_d.float().mean():.4f}, std={out_d.float().std():.4f}")

    # Reference decode correctness check
    x_d = a_d.float() + dt_bias.float()
    g_ref  = torch.exp(-torch.exp(A_log.float()) * F.softplus(x_d))   # [B,1,Hv]
    beta_ref = torch.sigmoid(b_d.float())                               # [B,1,Hv]
    gv_ratio = Hv // Hq
    q_exp = q_d.float().squeeze(1).repeat_interleave(gv_ratio, dim=1)  # [B,Hv,D]
    k_exp = k_d.float().squeeze(1).repeat_interleave(gv_ratio, dim=1)  # [B,Hv,D]
    v_f   = v_d.float().squeeze(1)                                      # [B,Hv,D]
    g_f   = g_ref.squeeze(1)                                            # [B,Hv]
    beta_f= beta_ref.squeeze(1)                                         # [B,Hv]
    ref_out = torch.zeros(B, Hv, D, device=device)
    ref_ns  = torch.zeros(B, Hv, D, D, device=device)
    for bi in range(B):
        for h in range(Hv):
            S_kv = state_d[bi, h].T.clone().float()   # [V,K] -> [K,V]
            g_h = g_f[bi, h]; beta_h = beta_f[bi, h]
            old_s = g_h * S_kv
            old_v = k_exp[bi, h] @ old_s
            dv    = beta_h * (v_f[bi, h] - old_v)
            S_kv  = old_s + k_exp[bi, h].unsqueeze(1) * dv.unsqueeze(0)
            ref_out[bi, h] = scale_val * (q_exp[bi, h] @ S_kv)
            ref_ns[bi, h]  = S_kv.T
    ref_out_bf16 = ref_out.unsqueeze(1).bfloat16()
    print(f"[decode] max abs output err vs ref: {(out_d.float() - ref_out_bf16.float()).abs().max():.3e}")
    print(f"[decode] max abs state  err vs ref: {(ns_d - ref_ns).abs().max():.3e}")

    # ---- Reference check ----
    Hsab = Hv
    q_exp = q.float().repeat_interleave(Hv // Hq, dim=1)    # [T, 8, 128]
    k_exp = k.float().repeat_interleave(Hv // Hq, dim=1)    # [T, 8, 128]
    a_f   = a_in.float()
    b_f   = b_in.float()
    g_all   = torch.exp(-torch.exp(A_log) * F.softplus(a_f + dt_bias))  # [T,8]
    beta_all = torch.sigmoid(b_f)  # [T,8]

    ref_out   = torch.zeros(T, Hsab, D, dtype=torch.float32, device=device)
    ref_state = torch.zeros(N, Hsab, D, D, dtype=torch.float32, device=device)

    for seq_idx in range(N):
        s_start = int(cu_seqlens[seq_idx].item())
        s_end   = int(cu_seqlens[seq_idx + 1].item())
        S_h = torch.zeros(Hsab, D, D, dtype=torch.float32, device=device)  # [H,V,K]
        for t in range(s_start, s_end):
            for h in range(Hsab):
                k_t   = k_exp[t, h]                # [K]
                v_t   = v[t, h].float()            # [V]
                q_t   = q_exp[t, h]                # [K]
                g_t   = g_all[t, h].item()
                beta_t = beta_all[t, h].item()
                S     = S_h[h]                     # [V, K]
                old_v = S @ k_t                    # [V]
                dv    = beta_t * (v_t - old_v)
                S_new = g_t * S + torch.outer(dv, k_t)
                S_h[h] = S_new
                ref_out[t, h] = (S_new @ q_t) * scale_val
        ref_state[seq_idx] = S_h

    max_err = (out.float() - ref_out).abs().max().item()
    print(f"Max abs error vs reference: {max_err:.4e}")
    print("PASS" if max_err < 0.1 else "FAIL — check kernel")
    print("Done!")
