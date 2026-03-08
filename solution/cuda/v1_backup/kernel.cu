#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

// Exact match of original activations to prevent ULP drift
__device__ __forceinline__ float softplus_exact(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ __forceinline__ float sigmoid_exact(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

template <int HEAD_SIZE>
__global__ void gated_delta_net_prefill_optimized_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ state,
    const float* __restrict__ A_log,
    const float* __restrict__ a,
    const float* __restrict__ dt_bias,
    const float* __restrict__ b,
    const int* __restrict__ cu_seqlens,
    float scale,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ new_state,
    int q_heads,
    int kv_heads
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x; // tid maps to the V dimension (v_idx)

    int seq_start = cu_seqlens[seq_idx];
    int seq_end   = cu_seqlens[seq_idx + 1];
    int seq_len   = seq_end - seq_start;
    if (seq_len <= 0) return;

    // Native GQA mapping
    int kv_group_size = q_heads / kv_heads;
    int kv_head_idx = head_idx / kv_group_size;

    constexpr int VEC_SIZE = 4;
    constexpr int NUM_VECS = HEAD_SIZE / VEC_SIZE;

    // Shared memory ONLY for Q and K (V is purely local now)
    extern __shared__ float shared_mem[];
    float* k_sh = shared_mem;
    float* q_sh = k_sh + HEAD_SIZE;
    
    // Aligned float4 pointers for vectorized math
    float4* k_sh_vec = reinterpret_cast<float4*>(k_sh);
    float4* q_sh_vec = reinterpret_cast<float4*>(q_sh);

    // Register State: Thread 'tid' (v_idx) owns state_global[v_idx, :]
    // This allows contiguous float4 loads from global memory!
    float4 r_state[NUM_VECS];

    if (state != nullptr) {
        size_t row_base = (size_t)seq_idx * q_heads * HEAD_SIZE * HEAD_SIZE +
                          (size_t)head_idx * HEAD_SIZE * HEAD_SIZE + 
                          (size_t)tid * HEAD_SIZE; // state[tid, 0]
        
        const float4* state_vec = reinterpret_cast<const float4*>(&state[row_base]);
        #pragma unroll
        for (int i = 0; i < NUM_VECS; ++i) {
            r_state[i] = state_vec[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < NUM_VECS; ++i) r_state[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Cache static head params
    float head_A_log = A_log[head_idx];
    float head_dt_bias = dt_bias[head_idx];

    for (int i = 0; i < seq_len; ++i) {
        int t = seq_start + i;
        
        size_t q_offset  = (size_t)t * q_heads * HEAD_SIZE + (size_t)head_idx * HEAD_SIZE + tid;
        size_t kv_offset = (size_t)t * kv_heads * HEAD_SIZE + (size_t)kv_head_idx * HEAD_SIZE + tid;

        // 1. Load Tokens
        float q_val = __bfloat162float(q[q_offset]);
        float k_val = __bfloat162float(k[kv_offset]);
        float v_val = __bfloat162float(v[kv_offset]); // v_val corresponds to v_idx (tid)!
        
        q_sh[tid] = q_val;
        k_sh[tid] = k_val;

        // 2. Scalars
        float a_val = a[t * q_heads + head_idx];
        float b_val = b[t * q_heads + head_idx];
        
        float sp = softplus_exact(a_val + head_dt_bias);
        float g = expf(-expf(head_A_log) * sp);
        float beta = sigmoid_exact(b_val);

        __syncthreads(); // Barrier for Q and K

        // 3. Compute old_v = dot(K, state_row)
        // Restored DOUBLE precision to exactly match original math
        double old_v_val = 0.0;
        #pragma unroll
        for (int j = 0; j < NUM_VECS; ++j) {
            float4 k_chunk = k_sh_vec[j];
            float4 s_chunk = r_state[j];
            old_v_val += (double)k_chunk.x * (double)s_chunk.x;
            old_v_val += (double)k_chunk.y * (double)s_chunk.y;
            old_v_val += (double)k_chunk.z * (double)s_chunk.z;
            old_v_val += (double)k_chunk.w * (double)s_chunk.w;
        }

        // 4. Update State and Compute Output
        float delta_v = beta * (v_val - (float)old_v_val);
        double out_accum = 0.0;

        #pragma unroll
        for (int j = 0; j < NUM_VECS; ++j) {
            float4 k_chunk = k_sh_vec[j];
            float4 q_chunk = q_sh_vec[j];
            float4 s_chunk = r_state[j];
            
            // Exact fmaf match: s = fmaf(k_val, delta_v, s * g)
            s_chunk.x = fmaf(k_chunk.x, delta_v, s_chunk.x * g);
            s_chunk.y = fmaf(k_chunk.y, delta_v, s_chunk.y * g);
            s_chunk.z = fmaf(k_chunk.z, delta_v, s_chunk.z * g);
            s_chunk.w = fmaf(k_chunk.w, delta_v, s_chunk.w * g);
            r_state[j] = s_chunk;

            out_accum += (double)q_chunk.x * (double)s_chunk.x;
            out_accum += (double)q_chunk.y * (double)s_chunk.y;
            out_accum += (double)q_chunk.z * (double)s_chunk.z;
            out_accum += (double)q_chunk.w * (double)s_chunk.w;
        }

        output[q_offset] = __float2bfloat16((float)(out_accum * (double)scale));

        __syncthreads(); // Guard shared memory for next sequence token
    }

    // Write final state using float4 coalesced stores
    size_t new_state_base = (size_t)seq_idx * q_heads * HEAD_SIZE * HEAD_SIZE +
                            (size_t)head_idx * HEAD_SIZE * HEAD_SIZE + 
                            (size_t)tid * HEAD_SIZE;
    float4* new_state_vec = reinterpret_cast<float4*>(&new_state[new_state_base]);
    #pragma unroll
    for (int i = 0; i < NUM_VECS; ++i) {
        new_state_vec[i] = r_state[i];
    }
}

extern "C" int launch_gated_delta_net_prefill_optimized(
    const void* q, const void* k, const void* v, const float* state,
    const float* A_log, const float* a, const float* dt_bias, const float* b,
    const int* cu_seqlens, float scale, void* output, float* new_state,
    int num_seqs, int q_heads, int kv_heads, int head_size, cudaStream_t stream
) {
    dim3 grid(num_seqs, q_heads, 1);
    dim3 block(head_size, 1, 1);
    int shared_mem = head_size * sizeof(float) * 2; 

    if (head_size == 128) {
        gated_delta_net_prefill_optimized_kernel<128><<<grid, block, shared_mem, stream>>>(
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v, state,
            A_log, a, dt_bias, b, cu_seqlens, scale, (__nv_bfloat16*)output, new_state, q_heads, kv_heads
        );
    } else if (head_size == 64) {
        gated_delta_net_prefill_optimized_kernel<64><<<grid, block, shared_mem, stream>>>(
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v, state,
            A_log, a, dt_bias, b, cu_seqlens, scale, (__nv_bfloat16*)output, new_state, q_heads, kv_heads
        );
    } else {
        return -1; 
    }
    return 0; 
}