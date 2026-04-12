#include "pairwise_mul.h"

namespace sorei::nn::layer {

constexpr int BLOCK_SIZE = 256;

__global__ void
pairwise_mul_fwd_kernel(const float* in, float* out, const int out_r, const int out_c) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_r * out_c)
        return;

    const int batch_idx = idx / out_r;
    const int feature_idx = idx % out_r;

    const int in_offset_a = batch_idx * 2 * out_r + feature_idx;
    const int in_offset_b = in_offset_a + out_r;

    out[batch_idx * out_r + feature_idx] = in[in_offset_a] * in[in_offset_b];
}

void PairwiseMul::forward() {
    auto& in = input_->data();
    auto& out = data();

    SOREI_CHECK(in.rows() % 2 == 0);
    SOREI_CHECK(in.cols() == out.cols());
    SOREI_CHECK(out.rows() == in.rows() / 2);

    SOREI_CHECK(in.data());
    SOREI_CHECK(out.data());

    const int blocks = cuda::ceil_div(out.size(), BLOCK_SIZE);
    pairwise_mul_fwd_kernel<<<blocks, BLOCK_SIZE>>>(in.data(), out.data(), out.rows(), out.cols());

    SOREI_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::nn::layer
