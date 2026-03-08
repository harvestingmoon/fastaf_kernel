#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h> 

extern "C" int launch_gated_delta_net_prefill_optimized(
    const void* q, const void* k, const void* v, const float* state,
    const float* A_log, const float* a, const float* dt_bias, const float* b,
    const int* cu_seqlens, float scale, void* output, float* new_state,
    int num_seqs, int q_heads, int kv_heads, int head_size, cudaStream_t stream
);

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

void kernel(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor state,
    torch::Tensor A_log, torch::Tensor a, torch::Tensor dt_bias, torch::Tensor b,
    torch::Tensor cu_seqlens, double scale, torch::Tensor output, torch::Tensor new_state
) {
    CHECK_CUDA(q); CHECK_CONTIGUOUS(q);
    CHECK_CUDA(k); CHECK_CONTIGUOUS(k);
    CHECK_CUDA(v); CHECK_CONTIGUOUS(v);
    CHECK_CUDA(cu_seqlens); CHECK_CONTIGUOUS(cu_seqlens);
    CHECK_CUDA(output); CHECK_CONTIGUOUS(output);
    CHECK_CUDA(new_state); CHECK_CONTIGUOUS(new_state);

    const int q_heads = q.size(1);
    const int kv_heads = k.size(1);
    const int head_size = q.size(2);
    const int num_seqs = cu_seqlens.size(0) - 1;

    TORCH_CHECK(q_heads % kv_heads == 0, "q_heads must be divisible by kv_heads for GQA");

    const float* state_ptr = nullptr;
    if (state.defined() && state.numel() > 0) {
        CHECK_CONTIGUOUS(state);
        state_ptr = state.data_ptr<float>();
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int launch_err = launch_gated_delta_net_prefill_optimized(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        state_ptr,
        A_log.data_ptr<float>(),
        a.data_ptr<float>(),
        dt_bias.data_ptr<float>(),
        b.data_ptr<float>(),
        cu_seqlens.data_ptr<int>(),
        static_cast<float>(scale),
        output.data_ptr(),
        new_state.data_ptr<float>(),
        num_seqs,
        q_heads,
        kv_heads,
        head_size,
        stream
    );

    TORCH_CHECK(launch_err == 0, "CUDA kernel launch failed. Unsupported head size.");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &kernel, "GDN prefill CUDA kernel");
}