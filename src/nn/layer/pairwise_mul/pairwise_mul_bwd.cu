#include "pairwise_mul.h"

namespace sorei::nn::layer {

constexpr int BLOCK_SIZE = 256;

__global__ void pairwise_mul_bwd_kernel(
    const float* in_d, float* in_g, const float* out_g, const int out_r, const int out_c
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_r * out_c)
        return;

    const int batch_idx = idx / out_r;
    const int feature_idx = idx % out_r;

    const float grad = out_g[batch_idx * out_r + feature_idx];

    const int in_offset_a = batch_idx * 2 * out_r + feature_idx;
    const int in_offset_b = in_offset_a + out_r;

    in_g[in_offset_a] += grad * in_d[in_offset_b];
    in_g[in_offset_b] += grad * in_d[in_offset_a];
}

void PairwiseMul::backward() {
    auto& in_g = input_->grad();
    auto& out_g = grad();
    auto& in = input_->data();

    if (!input_->requires_grad())
        return;

    CHECK(in.rows() % 2 == 0);
    CHECK(in.cols() == out_g.cols());
    CHECK(out_g.rows() == in.rows() / 2);

    CHECK(in.data());
    CHECK(out_g.data());

    const int blocks = cuda::ceil_div(out_g.size(), BLOCK_SIZE);
    pairwise_mul_bwd_kernel<<<blocks, BLOCK_SIZE>>>(
        in.data(), in_g.data(), out_g.data(), out_g.rows(), out_g.cols()
    );

    CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::nn::layer
