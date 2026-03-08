/*
 * wrapper.cpp — GDN decode, destination-passing style (DPS).
 *
 * Kernel entry point: `kernel`
 * Signature follows gdn_decode_qk4_v8_d128_k_last definition.
 *
 * I/O overhead removed vs naïve approach:
 *  • No repeat_interleave  — GVA (Hk=4→Hv=8) handled inside CUDA kernel.
 *  • No host-side a/b type conversion — kernel reads bfloat16 directly.
 *  • Minimal .contiguous() calls — only where strictly required.
 *  • scale=0/None handled here once, not inside Python on every call.
 *  • state=None handled once — kernel always receives a valid pointer.
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

extern "C" int launch_gdn_decode(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const float*         state,
    const float*         A_log,
    const __nv_bfloat16* a,
    const float*         dt_bias,
    const __nv_bfloat16* b,
    float                scale,
    __nv_bfloat16*       output,
    float*               new_state,
    int batch_size,
    int num_v_heads,
    int num_qk_heads,
    int head_size,
    cudaStream_t stream
);

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")

void kernel(
    torch::Tensor q,           // [B, 1, Hq, D]  bfloat16
    torch::Tensor k,           // [B, 1, Hk, D]  bfloat16
    torch::Tensor v,           // [B, 1, Hv, D]  bfloat16
    torch::Tensor state,       // [B, Hv, D, D]  float32  (k-last) or empty
    torch::Tensor A_log,       // [Hv]            float32
    torch::Tensor a,           // [B, 1, Hv]      bfloat16
    torch::Tensor dt_bias,     // [Hv]            float32
    torch::Tensor b,           // [B, 1, Hv]      bfloat16
    double        scale_d,     // scalar (0 = default 1/sqrt(D))
    torch::Tensor output,      // [B, 1, Hv, D]  bfloat16  (pre-allocated, DPS)
    torch::Tensor new_state    // [B, Hv, D, D]  float32   (pre-allocated, DPS)
) {
    CHECK_CUDA(q); CHECK_CUDA(k); CHECK_CUDA(v);
    CHECK_CUDA(A_log); CHECK_CUDA(a); CHECK_CUDA(dt_bias); CHECK_CUDA(b);
    CHECK_CUDA(output); CHECK_CUDA(new_state);

    TORCH_CHECK(q.dim() == 4, "q must be [B, 1, Hq, D]");
    TORCH_CHECK(v.dim() == 4, "v must be [B, 1, Hv, D]");

    const int batch_size   = (int)q.size(0);
    const int num_qk_heads = (int)q.size(2);
    const int num_v_heads  = (int)v.size(2);
    const int head_size    = (int)q.size(3);

    TORCH_CHECK(head_size == 128, "Only head_size=128 is supported");
    TORCH_CHECK(num_v_heads % num_qk_heads == 0,
                "num_v_heads must be divisible by num_qk_heads");

    // Resolve scale
    float scale = (float)scale_d;
    if (scale <= 0.0f) scale = 1.0f / sqrtf((float)head_size);

    // Contiguous inputs — no-op if already contiguous
    auto q_c       = q.contiguous();
    auto k_c       = k.contiguous();
    auto v_c       = v.contiguous();
    auto A_log_c   = A_log.contiguous();
    auto dt_bias_c = dt_bias.contiguous();
    // a, b: bfloat16 [B, 1, Hv] — kernel reads bf16 directly, just ensure contiguous
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();

    // State: if undefined or empty, pass a zero tensor (rare first-call path)
    const float* state_ptr = nullptr;
    torch::Tensor state_zero;
    if (state.defined() && state.numel() > 0) {
        auto sc = state.is_contiguous() ? state : state.contiguous();
        state   = sc;  // keep alive
        state_ptr = state.data_ptr<float>();
    } else {
        state_zero = torch::zeros(
            {batch_size, num_v_heads, head_size, head_size},
            torch::dtype(torch::kFloat32).device(q.device()));
        state_ptr = state_zero.data_ptr<float>();
    }

    auto output_c    = output.is_contiguous()    ? output    : output.contiguous();
    auto new_state_c = new_state.is_contiguous() ? new_state : new_state.contiguous();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int err = launch_gdn_decode(
        reinterpret_cast<const __nv_bfloat16*>(q_c.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_c.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_c.data_ptr()),
        state_ptr,
        A_log_c.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(a_c.data_ptr()),
        dt_bias_c.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(b_c.data_ptr()),
        scale,
        reinterpret_cast<__nv_bfloat16*>(output_c.data_ptr()),
        new_state_c.data_ptr<float>(),
        batch_size,
        num_v_heads,
        num_qk_heads,
        head_size,
        stream
    );

    TORCH_CHECK(err == 0, "launch_gdn_decode failed with CUDA error ", err);

    // Copy back only if we had to materialise contiguous copies (rare)
    if (!output.is_contiguous())    output.copy_(output_c);
    if (!new_state.is_contiguous()) new_state.copy_(new_state_c);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &kernel, "GDN decode CUDA kernel (DPS)");
}