/*
 * Gated Delta Net CUDA kernels — CC 8.6+ (Ampere).
 *
 * DECODE  (gdn_decode_kernel)
 *   Grid (B, Hv), Block (HEAD_SIZE).  Thread tid owns V-row tid of state.
 *   smem: state_sh[V][K+1] + k_sh[K] + q_sh[K] ≈ 65.5 KB (limit: 100 KB).
 *   Coalesced I/O: all threads load/store state one row at a time (stride-1).
 *   GVA: qk_head_idx = v_head_idx / GV_RATIO (compile-time constant).
 *   a, b: bfloat16 [B, Hv].  state=nullptr → zero-initialised in kernel.
 *
 * PREFILL (gated_delta_net_prefill_kernel_opt)
 *   Grid (num_seqs, num_sab_heads), Block (HEAD_SIZE).
 *   Thread tid owns state row tid.  GVA handled natively (no host expansion).
 *   a, b: bfloat16 [T, v_heads] — each value is shared across HEAD_SIZE threads.
 *   One __syncthreads() per token iteration (covers q_sh/k_sh write-before-read).
 *   state=nullptr → zero-initialised in registers.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

// float32 dot product over float4 register arrays
template <int HEAD_SIZE>
__device__ __forceinline__ float dot_float4(const float4* __restrict__ a,
                                            const float4* __restrict__ b) {
    constexpr int NUM_VECS = HEAD_SIZE / 4;
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < NUM_VECS; ++i) {
        const float4 av = a[i];
        const float4 bv = b[i];
        acc = fmaf(av.x, bv.x, acc);
        acc = fmaf(av.y, bv.y, acc);
        acc = fmaf(av.z, bv.z, acc);
        acc = fmaf(av.w, bv.w, acc);
    }
    return acc;
}

__device__ __forceinline__ float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ __forceinline__ float sigmoid(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

// ─────────────────────────────────────────────────────────────────────────────
// DECODE KERNEL
// ─────────────────────────────────────────────────────────────────────────────
template <int HEAD_SIZE, int GV_RATIO>
__global__ void __launch_bounds__(HEAD_SIZE)
gdn_decode_kernel(
    const __nv_bfloat16* __restrict__ q,          // [B, Hq, D]
    const __nv_bfloat16* __restrict__ k,          // [B, Hk, D]
    const __nv_bfloat16* __restrict__ v,          // [B, Hv, D]
    const float*         __restrict__ state,      // [B, Hv, D, D] k-last [V,K]
    const float*         __restrict__ A_log,      // [Hv]
    const __nv_bfloat16* __restrict__ a,          // [B, Hv]
    const float*         __restrict__ dt_bias,    // [Hv]
    const __nv_bfloat16* __restrict__ b,          // [B, Hv]
    float                             scale,
    __nv_bfloat16*       __restrict__ output,     // [B, Hv, D]
    float*               __restrict__ new_state,  // [B, Hv, D, D] k-last [V,K]
    int Hq, int Hv
) {
    // smem layout constants
    constexpr int SMEM_STRIDE = HEAD_SIZE + 1;     // 129  (odd → no bank conflicts)
    constexpr int STATE_SH_SZ = HEAD_SIZE * SMEM_STRIDE;  // floats for state_sh
    constexpr int NUM_VECS    = HEAD_SIZE / 4;

    const int batch_idx   = blockIdx.x;
    const int v_head_idx  = blockIdx.y;
    const int qk_head_idx = v_head_idx / GV_RATIO;
    const int tid         = threadIdx.x;  // = V-row index this thread owns

    // ── shared memory pointers ───────────────────────────────────────────────
    extern __shared__ float smem[];                       // dynamic
    float* __restrict__ state_sh = smem;                  // [HEAD_SIZE][HEAD_SIZE+1]
    float* __restrict__ k_sh     = smem + STATE_SH_SZ;    // [HEAD_SIZE]
    float* __restrict__ q_sh     = k_sh + HEAD_SIZE;      // [HEAD_SIZE]

    // ── load q, k into shared (coalesced) ────────────────────────────────────
    {
        const size_t q_off = (size_t)batch_idx * Hq * HEAD_SIZE
                           + (size_t)qk_head_idx * HEAD_SIZE + tid;
        const size_t k_off = (size_t)batch_idx * (Hv / GV_RATIO) * HEAD_SIZE
                           + (size_t)qk_head_idx * HEAD_SIZE + tid;
        q_sh[tid] = __bfloat162float(__ldg(q + q_off));
        k_sh[tid] = __bfloat162float(__ldg(k + k_off));
    }

    // ── load per-head scalars (uniform across block → L1 broadcast) ──────────
    const float head_A_log   = __ldg(A_log   + v_head_idx);
    const float head_dt_bias = __ldg(dt_bias + v_head_idx);

    const size_t ab_off = (size_t)batch_idx * Hv + v_head_idx;
    const float a_val = __bfloat162float(__ldg(a + ab_off));
    const float b_val = __bfloat162float(__ldg(b + ab_off));

    const float g    = expf(-expf(head_A_log) * softplus(a_val + head_dt_bias));
    const float beta = sigmoid(b_val);

    // ── load v[tid] — coalesced ───────────────────────────────────────────────
    const size_t v_off = (size_t)batch_idx * Hv * HEAD_SIZE
                       + (size_t)v_head_idx * HEAD_SIZE + tid;
    const float v_val = __bfloat162float(__ldg(v + v_off));

    // ── load state into shared memory (COALESCED global reads) ───────────────
    // state global layout: [B, Hv, V=HEAD_SIZE, K=HEAD_SIZE]  (k-last)
    //   flat addr of state[b][hv][v][k] = base + v*HEAD_SIZE + k
    // All threads load state_sh[v][tid] = state[v][tid]: stride-1 → coalesced.
    // When state is nullptr (first decode step), zero-init directly in smem.
    {
        const size_t base = ((size_t)batch_idx * Hv + v_head_idx)
                          * (size_t)HEAD_SIZE * HEAD_SIZE;
        if (state != nullptr) {
#pragma unroll 8
            for (int v = 0; v < HEAD_SIZE; ++v)
                state_sh[v * SMEM_STRIDE + tid] =
                    __ldg(state + base + (size_t)v * HEAD_SIZE + tid);
        } else {
#pragma unroll 8
            for (int v = 0; v < HEAD_SIZE; ++v)
                state_sh[v * SMEM_STRIDE + tid] = 0.f;
        }
    }

    // Single barrier covers k_sh, q_sh AND state_sh
    __syncthreads();

    const float4* k_sh4 = reinterpret_cast<const float4*>(k_sh);
    const float4* q_sh4 = reinterpret_cast<const float4*>(q_sh);

    // ── read state row tid from smem into registers ───────────────────────────
    // state_sh[v][tid] at smem offset  v * SMEM_STRIDE + tid
    // Thread tid needs: state_sh[tid][0..HEAD_SIZE-1]
    //   = smem[tid * SMEM_STRIDE + 0..127]  (consecutive in the padded row)
    // bank(tid, k) = (tid * SMEM_STRIDE + k) % 32 = (tid + k) % 32 (since SMEM_STRIDE=129=128+1)
    //   For full warp: bank(j,k) = (j+k)%32 → all 32 banks hit once → conflict-free
    float* my_row = state_sh + (size_t)tid * SMEM_STRIDE;
    float4 row_state[NUM_VECS];
#pragma unroll
    for (int i = 0; i < NUM_VECS; ++i) {
        const int base_k = i * 4;
        row_state[i] = make_float4(my_row[base_k],
                                   my_row[base_k + 1],
                                   my_row[base_k + 2],
                                   my_row[base_k + 3]);
    }
    // ── step 1: decay state in-place ─────────────────────────────────────────
#pragma unroll
    for (int i = 0; i < NUM_VECS; ++i) {
        float4 s = row_state[i];
        s.x *= g; s.y *= g; s.z *= g; s.w *= g;
        row_state[i] = s;
    }

    // ── step 2: old_v = dot(k, decayed_row)  (float32) ───────────────────────
    const float old_v = dot_float4<HEAD_SIZE>(k_sh4, row_state);

    // ── step 3: delta_v ───────────────────────────────────────────────────────
    const float delta_v = beta * (v_val - old_v);

    // ── step 4: rank-1 update + output dot (fused, float32) ──────────────────
    float out_acc = 0.0f;
#pragma unroll
    for (int i = 0; i < NUM_VECS; ++i) {
        const float4 kv = k_sh4[i];
        const float4 qv = q_sh4[i];
        float4 s = row_state[i];

        s.x = fmaf(kv.x, delta_v, s.x);
        s.y = fmaf(kv.y, delta_v, s.y);
        s.z = fmaf(kv.z, delta_v, s.z);
        s.w = fmaf(kv.w, delta_v, s.w);
        row_state[i] = s;

        out_acc = fmaf(qv.x, s.x, fmaf(qv.y, s.y,
                  fmaf(qv.z, s.z, fmaf(qv.w, s.w, out_acc))));
    }

    // ── write output (coalesced) ──────────────────────────────────────────────
    {
        const size_t out_off = (size_t)batch_idx * Hv * HEAD_SIZE
                             + (size_t)v_head_idx * HEAD_SIZE + tid;
        output[out_off] = __float2bfloat16(out_acc * scale);
    }

    // ── write updated row back to smem ────────────────────────────────────────
#pragma unroll
    for (int i = 0; i < NUM_VECS; ++i) {
        const int base_k     = i * 4;
        float* dst           = my_row + base_k;   // my_row = state_sh + tid*SMEM_STRIDE
        dst[0] = row_state[i].x;
        dst[1] = row_state[i].y;
        dst[2] = row_state[i].z;
        dst[3] = row_state[i].w;
    }

    // Barrier before coalesced write-back to global new_state
    __syncthreads();

    // ── write new_state (COALESCED global writes) ─────────────────────────────
    // Symmetric to the coalesced load:
    //   new_state[v][tid] = state_sh[v][tid]
    //   at step v, warp threads write new_state_base + v*HEAD_SIZE + 0..31 → coalesced
    {
        const size_t base = ((size_t)batch_idx * Hv + v_head_idx)
                          * (size_t)HEAD_SIZE * HEAD_SIZE;
#pragma unroll 8
        for (int v = 0; v < HEAD_SIZE; ++v)
            new_state[base + (size_t)v * HEAD_SIZE + tid] = state_sh[v * SMEM_STRIDE + tid];
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DECODE LAUNCHER
// ──────────────────────────────────────────────────────────────────────────────
extern "C" int launch_gdn_decode(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const float*         state,          // nullable → zero-init done in wrapper
    const float*         A_log,
    const __nv_bfloat16* a,
    const float*         dt_bias,
    const __nv_bfloat16* b,
    float                scale,
    __nv_bfloat16*       output,
    float*               new_state,
    int batch_size,
    int num_v_heads,                     // Hv  (e.g. 8)
    int num_qk_heads,                    // Hq = Hk (e.g. 4)
    int head_size,
    cudaStream_t stream
) {
    const int gv_ratio = num_v_heads / num_qk_heads;  // 2
    dim3 grid(batch_size, num_v_heads, 1);
    dim3 block(head_size, 1, 1);

    // Dynamic shared memory:
    //   state_sh : head_size * (head_size+1) floats  (PAD=1 avoids bank conflicts)
    //   k_sh     : head_size floats
    //   q_sh     : head_size floats
    const int smem = (head_size * (head_size + 1) + 2 * head_size) * (int)sizeof(float);

    if (head_size == 128 && gv_ratio == 2) {
        // Allow >48 KB dynamic shared memory per block (Ampere: up to 100 KB)
        cudaError_t attr_err = cudaFuncSetAttribute(
            (const void*)gdn_decode_kernel<128, 2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem);
        if (attr_err != cudaSuccess) return (int)attr_err;

        gdn_decode_kernel<128, 2><<<grid, block, smem, stream>>>(
            q, k, v, state, A_log, a, dt_bias, b, scale,
            output, new_state, num_qk_heads, num_v_heads);
    } else {
        return (int)cudaErrorInvalidValue;
    }
    return (int)cudaGetLastError();
}

// ─────────────────────────────────────────────────────────────────────────────
// PREFILL KERNEL
//
// Grid  : (num_seqs, num_sab_heads)   Block : (HEAD_SIZE)
// Thread tid owns state row tid (V-row index).
//
// GVA handled natively: q/k/v are indexed with their *original* head counts.
// a, b are bfloat16 [T, v_heads] — indexed with v_heads stride.
// One __syncthreads() per iteration (top of loop, after writing q_sh/k_sh).
// ─────────────────────────────────────────────────────────────────────────────
template <int HEAD_SIZE>
__global__ __launch_bounds__(HEAD_SIZE)
void gated_delta_net_prefill_kernel_opt(
    const __nv_bfloat16* __restrict__ q,         // [T, q_heads, D]
    const __nv_bfloat16* __restrict__ k,         // [T, k_heads, D]
    const __nv_bfloat16* __restrict__ v,         // [T, v_heads, D]
    const float*         __restrict__ state,     // [S, sab_heads, D, D] or nullptr
    const float*         __restrict__ A_log,     // [sab_heads]
    const __nv_bfloat16* __restrict__ a,         // [T, v_heads]  bf16
    const float*         __restrict__ dt_bias,   // [sab_heads]
    const __nv_bfloat16* __restrict__ b,         // [T, v_heads]  bf16
    const int*           __restrict__ cu_seqlens,
    float scale,
    __nv_bfloat16*       __restrict__ output,    // [T, sab_heads, D]
    float*               __restrict__ new_state, // [S, sab_heads, D, D]
    int q_heads,
    int k_heads,
    int v_heads,
    int num_sab_heads
) {
    constexpr int NUM_VECS = HEAD_SIZE / 4;

    const int seq_idx      = blockIdx.x;
    const int sab_head_idx = blockIdx.y;
    const int tid          = threadIdx.x;

    const int seq_start = __ldg(cu_seqlens + seq_idx);
    const int seq_end   = __ldg(cu_seqlens + seq_idx + 1);
    const int seq_len   = seq_end - seq_start;
    if (seq_len <= 0) return;

    // GVA head mappings
    const int q_head_idx = sab_head_idx / (num_sab_heads / q_heads);
    const int k_head_idx = sab_head_idx / (num_sab_heads / k_heads);
    const int v_head_idx = sab_head_idx / (num_sab_heads / v_heads);

    extern __shared__ float shared_mem[];
    float* __restrict__ k_sh = shared_mem;
    float* __restrict__ q_sh = k_sh + HEAD_SIZE;

    const float4* k_sh_vec = reinterpret_cast<const float4*>(k_sh);
    const float4* q_sh_vec = reinterpret_cast<const float4*>(q_sh);

    // Per-head constants (uniform across block → L1 broadcast)
    const float head_A_log   = __ldg(A_log   + sab_head_idx);
    const float head_dt_bias = __ldg(dt_bias + sab_head_idx);

    // Load initial state row into registers
    float4 row_state[NUM_VECS];
    if (state != nullptr) {
        const size_t state_row_base =
            (size_t)seq_idx      * num_sab_heads * HEAD_SIZE * HEAD_SIZE +
            (size_t)sab_head_idx * HEAD_SIZE * HEAD_SIZE +
            (size_t)tid          * HEAD_SIZE;
        const float4* src = reinterpret_cast<const float4*>(state + state_row_base);
#pragma unroll
        for (int i = 0; i < NUM_VECS; ++i) row_state[i] = src[i];
    } else {
#pragma unroll
        for (int i = 0; i < NUM_VECS; ++i)
            row_state[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    for (int i = 0; i < seq_len; ++i) {
        const int t = seq_start + i;

        // Write q, k into smem — all threads participate
        q_sh[tid] = __bfloat162float(__ldg(q + (size_t)t * q_heads * HEAD_SIZE
                                               + (size_t)q_head_idx * HEAD_SIZE + tid));
        k_sh[tid] = __bfloat162float(__ldg(k + (size_t)t * k_heads * HEAD_SIZE
                                               + (size_t)k_head_idx * HEAD_SIZE + tid));

        // Load private scalars (no smem dependency)
        const float v_val = __bfloat162float(__ldg(v + (size_t)t * v_heads * HEAD_SIZE
                                                       + (size_t)v_head_idx * HEAD_SIZE + tid));
        // a/b are [T, v_heads] bf16 — one value per head (same for all D threads)
        const float a_val = __bfloat162float(__ldg(a + (size_t)t * v_heads + v_head_idx));
        const float b_val = __bfloat162float(__ldg(b + (size_t)t * v_heads + v_head_idx));

        const float g    = expf(-expf(head_A_log) * softplus(a_val + head_dt_bias));
        const float beta = sigmoid(b_val);

        // Barrier: wait for all threads to finish writing q_sh/k_sh
        __syncthreads();

        // Decay state row in-place
#pragma unroll
        for (int j = 0; j < NUM_VECS; ++j) {
            float4 s = row_state[j];
            s.x *= g; s.y *= g; s.z *= g; s.w *= g;
            row_state[j] = s;
        }

        // old_v = dot(k, decayed_row)
        const float old_v   = dot_float4<HEAD_SIZE>(k_sh_vec, row_state);
        const float delta_v = beta * (v_val - old_v);

        // Rank-1 update + output dot (fused)
        float out_acc = 0.f;
#pragma unroll
        for (int j = 0; j < NUM_VECS; ++j) {
            const float4 kv = k_sh_vec[j];
            const float4 qv = q_sh_vec[j];
            float4 s = row_state[j];
            s.x = fmaf(kv.x, delta_v, s.x);
            s.y = fmaf(kv.y, delta_v, s.y);
            s.z = fmaf(kv.z, delta_v, s.z);
            s.w = fmaf(kv.w, delta_v, s.w);
            row_state[j] = s;
            out_acc = fmaf(qv.x, s.x, fmaf(qv.y, s.y,
                      fmaf(qv.z, s.z, fmaf(qv.w, s.w, out_acc))));
        }

        output[(size_t)t * num_sab_heads * HEAD_SIZE
               + (size_t)sab_head_idx * HEAD_SIZE + tid] =
            __float2bfloat16(out_acc * scale);
        // No trailing __syncthreads() needed: next iteration's top barrier suffices
    }

    // Write final state row back (coalesced float4 stores)
    const size_t new_state_row_base =
        (size_t)seq_idx      * num_sab_heads * HEAD_SIZE * HEAD_SIZE +
        (size_t)sab_head_idx * HEAD_SIZE * HEAD_SIZE +
        (size_t)tid          * HEAD_SIZE;
    float4* dst = reinterpret_cast<float4*>(new_state + new_state_row_base);
#pragma unroll
    for (int i = 0; i < NUM_VECS; ++i) dst[i] = row_state[i];
}

extern "C" int launch_gated_delta_net_prefill_optimized(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const float*         state,
    const float*         A_log,
    const __nv_bfloat16* a,        // bf16 [T, v_heads]
    const float*         dt_bias,
    const __nv_bfloat16* b,        // bf16 [T, v_heads]
    const int*           cu_seqlens,
    float                scale,
    __nv_bfloat16*       output,
    float*               new_state,
    int num_seqs,
    int q_heads,
    int k_heads,
    int v_heads,
    int num_sab_heads,
    int head_size,
    cudaStream_t stream
) {
    dim3 grid(num_seqs, num_sab_heads, 1);
    dim3 block(head_size, 1, 1);
    const int shared_mem = 2 * head_size * (int)sizeof(float);

    cudaError_t err;
    if (head_size == 128) {
        err = cudaFuncSetAttribute(
            gated_delta_net_prefill_kernel_opt<128>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
        if (err != cudaSuccess) return (int)err;
        gated_delta_net_prefill_kernel_opt<128><<<grid, block, shared_mem, stream>>>(
            q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale,
            output, new_state, q_heads, k_heads, v_heads, num_sab_heads);
    } else if (head_size == 64) {
        err = cudaFuncSetAttribute(
            gated_delta_net_prefill_kernel_opt<64>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
        if (err != cudaSuccess) return (int)err;
        gated_delta_net_prefill_kernel_opt<64><<<grid, block, shared_mem, stream>>>(
            q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale,
            output, new_state, q_heads, k_heads, v_heads, num_sab_heads);
    } else {
        return (int)cudaErrorInvalidValue;
    }
    return (int)cudaGetLastError();
}

// ──────────────────────────────────────────────────────────────────────────────
// ALIAS expected by binding.py  (launch_gated_delta_net_prefill_raw)
// The kernel handles GVA natively. a/b are bf16 [T, v_heads].
// ──────────────────────────────────────────────────────────────────────────────
extern "C" int launch_gated_delta_net_prefill_raw(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const float*         state,
    const float*         A_log,
    const __nv_bfloat16* a,        // bf16 [T, v_heads]
    const float*         dt_bias,
    const __nv_bfloat16* b,        // bf16 [T, v_heads]
    const int*           cu_seqlens,
    float                scale,
    __nv_bfloat16*       output,
    float*               new_state,
    int num_seqs,
    int q_heads,
    int k_heads,
    int v_heads,
    int num_sab_heads,
    int head_size,
    cudaStream_t stream
) {
    return launch_gated_delta_net_prefill_optimized(
        q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale,
        output, new_state,
        num_seqs, q_heads, k_heads, v_heads, num_sab_heads, head_size, stream);
}
