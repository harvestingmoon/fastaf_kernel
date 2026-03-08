/*
 * CUDA Kernel for Gated Delta Net Prefill (k-last layout)
 *
 * This kernel processes each (sequence, head) independently.
 * State is kept in shared memory as float32 for numerical stability and updated token by token.
 * All intermediate arithmetic is performed in float32 for numerical stability.

 * Expected input shapes (all tensors must be contiguous):
 *   q, k, v : [total_seq_len, num_sab_heads, head_size] (bfloat16)
 *   state (optional) : [num_seqs, num_sab_heads, head_size, head_size] (float32) layout [V,K]
 *   A_log, dt_bias : [num_sab_heads] (float32)
 *   a, b : [total_seq_len, num_sab_heads] (float32)
 *   cu_seqlens : [num_seqs+1] (int32)
 *   output : [total_seq_len, num_sab_heads, head_size] (bfloat16)
 *   new_state : [num_seqs, num_sab_heads, head_size, head_size] (float32) layout [V,K]
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>

// Helper: softplus with overflow protection
__device__ float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

// Helper: sigmoid
__device__ float sigmoid(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

__global__ void gated_delta_net_prefill_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ state,           // may be nullptr
    const float* __restrict__ A_log,
    const float* __restrict__ a,
    const float* __restrict__ dt_bias,
    const float* __restrict__ b,
    const int* __restrict__ cu_seqlens,
    float scale,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ new_state,
    int num_sab_heads,
    int head_size
) {
    // Block index: (seq_idx, head_idx)
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;               // 0 .. head_size-1 (head_size = 128)

    int seq_start = cu_seqlens[seq_idx];
    int seq_end   = cu_seqlens[seq_idx + 1];
    int seq_len = seq_end - seq_start;
    if (seq_len <= 0) return;

    const int H = head_size;

    // Shared memory layout:
    //   state_sh   : float[H][H]
    //   k_sh       : float[H]              (512 B)
    //   v_sh       : float[H]              (512 B)
    //   q_sh       : float[H]              (512 B)
    //   old_v_sh   : float[H]              (512 B)
    //   new_v_sh   : float[H]              (512 B)
    //   g_val      : float                  (4 B)
    //   beta_val   : float                  (4 B)
    extern __shared__ char shared_mem[];
    float*  state_sh   = (float*)shared_mem;
    float*  k_sh       = state_sh + H * H;
    float*  v_sh       = k_sh + H;
    float*  q_sh       = v_sh + H;
    float*  old_v_sh   = q_sh + H;
    float*  new_v_sh   = old_v_sh + H;
    float*  g_ptr      = new_v_sh + H;
    float*  beta_ptr   = g_ptr + 1;

    // ----- Initialise state matrix from global (or zero) -----
    if (state != nullptr) {
        int k_idx = tid;
        size_t state_base = seq_idx * num_sab_heads * H * H +
                            head_idx * H * H;
        for (int v_idx = 0; v_idx < H; ++v_idx) {
            float val = state[state_base + v_idx * H + k_idx];
            state_sh[k_idx * H + v_idx] = val;
        }
    } else {
        // Zero initialisation
        int k_idx = tid;
        for (int v_idx = 0; v_idx < H; ++v_idx) {
            state_sh[k_idx * H + v_idx] = 0.0f;
        }
    }
    __syncthreads();
    for (int i = 0; i < seq_len; ++i) {
        int t = seq_start + i;                    
        if (tid == 0) {
            float A_log_val  = A_log[head_idx];
            float a_val      = a[t * num_sab_heads + head_idx];
            float dt_bias_val= dt_bias[head_idx];
            float b_val      = b[t * num_sab_heads + head_idx];

            float x = a_val + dt_bias_val;
            float sp = softplus(x);
            float exp_A = expf(A_log_val);
            float exponent = exp_A * sp;
            float g = expf(-exponent);             // g = exp(-exp(A_log)*softplus(a+dt_bias))
            float beta = sigmoid(b_val);

            *g_ptr = g;
            *beta_ptr = beta;
        }

        // ---- Load k, v, q vectors (coalesced) ----
        size_t token_head_offset = t * num_sab_heads * H + head_idx * H + tid;
        float k_elem = __bfloat162float(k[token_head_offset]);
        float v_elem = __bfloat162float(v[token_head_offset]);
        float q_elem = __bfloat162float(q[token_head_offset]);
        k_sh[tid] = k_elem;
        v_sh[tid] = v_elem;
        q_sh[tid] = q_elem;
        __syncthreads();

        // 1. Scale state by g (elementwise, per row)
        float g = *g_ptr;
        int row = tid;
        for (int col = 0; col < H; ++col) {
            float s = state_sh[row * H + col];
            s *= g;
            state_sh[row * H + col] = s;
        }
        __syncthreads();

        // 2. Compute old_v = k @ state   (result per column)
        int col = tid;
        double sum = 0.0;
        for (int r = 0; r < H; ++r) {
            sum += static_cast<double>(k_sh[r]) * static_cast<double>(state_sh[r * H + col]);
        }
        old_v_sh[col] = static_cast<float>(sum);
        __syncthreads();

        // 3. Compute new_v = beta * v + (1-beta) * old_v
        float beta = *beta_ptr;
        new_v_sh[col] = beta * v_sh[col] + (1.0f - beta) * old_v_sh[col];
        __syncthreads();

        // 4. Update state in cancellation-resistant form:
        //    state += k^T @ (new_v - old_v) = state + k^T @ (beta * (v - old_v))
        row = tid;
        float k_val = k_sh[row];
        for (int c = 0; c < H; ++c) {
            float old_v = old_v_sh[c];
            float v_cur = v_sh[c];
            float delta_v = beta * (v_cur - old_v);
            float s = state_sh[row * H + c];
            s = fmaf(k_val, delta_v, s);
            state_sh[row * H + c] = s;
        }
        __syncthreads();

        // 5. Compute output = scale * q @ state   (per column)
        col = tid;
        double out_sum = 0.0;
        for (int r = 0; r < H; ++r) {
            out_sum += static_cast<double>(q_sh[r]) * static_cast<double>(state_sh[r * H + col]);
        }
        out_sum *= static_cast<double>(scale);
        output[t * num_sab_heads * H + head_idx * H + col] = __float2bfloat16(static_cast<float>(out_sum));
        __syncthreads();   // ensure next iteration sees consistent state
    }

    // ----- Write final state back to global (transpose back to [V,K]) -----
    int k_idx = tid;
    size_t new_state_base = seq_idx * num_sab_heads * H * H +
                            head_idx * H * H;
    for (int v_idx = 0; v_idx < H; ++v_idx) {
        float val = state_sh[k_idx * H + v_idx];
        new_state[new_state_base + v_idx * H + k_idx] = val;
    }
}

extern "C" int launch_gated_delta_net_prefill_raw(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const float* state,
    const float* A_log,
    const float* a,
    const float* dt_bias,
    const float* b,
    const int* cu_seqlens,
    float scale,
    __nv_bfloat16* output,
    float* new_state,
    int num_seqs,
    int num_sab_heads,
    int head_size
) {
    dim3 grid(num_seqs, num_sab_heads, 1);
    dim3 block(head_size, 1, 1);
    int shared_mem = head_size * head_size * 4 + 5 * head_size * 4 + 8;

    cudaError_t err = cudaFuncSetAttribute(
        gated_delta_net_prefill_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem
    );
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    gated_delta_net_prefill_kernel<<<grid, block, shared_mem>>>(
        q,
        k,
        v,
        state,
        A_log,
        a,
        dt_bias,
        b,
        cu_seqlens,
        scale,
        output,
        new_state,
        num_sab_heads,
        head_size
    );

    err = cudaGetLastError();
    return static_cast<int>(err);
}