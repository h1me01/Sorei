#include "select.h"

namespace sorei::nn::layer {

constexpr int BLOCK_SIZE = 256;

__global__ void select_bwd_kernel(
    float* in_g,
    const float* out_g,
    const int* indices,
    const int in_r,
    const int out_r,
    const int out_c
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_c * out_r)
        return;

    const int batch_idx = idx / out_r;
    const int out_idx = idx % out_r;

    const int bucket = indices[batch_idx];
    const int in_offset = in_r * batch_idx + out_r * bucket + out_idx;

    const int out_offset = out_r * batch_idx + out_idx;
    in_g[in_offset] += out_g[out_offset];
}

void Select::backward() {
    auto& in_g = input_->grad();
    auto& out_g = grad();
    auto& indices = bucket_->data();

    if (!input_->requires_grad())
        return;

    CHECK(indices.rows() == 1);
    CHECK(in_g.cols() == out_g.cols());
    CHECK(out_g.cols() == indices.cols());
    CHECK(in_g.rows() % out_g.rows() == 0);

    CHECK(in_g.data());
    CHECK(out_g.data());
    CHECK(indices.data());

    const int blocks = cuda::ceil_div(out_g.size(), BLOCK_SIZE);
    select_bwd_kernel<<<blocks, BLOCK_SIZE>>>(
        in_g.data(), out_g.data(), indices.data(), in_g.rows(), out_g.rows(), out_g.cols()
    );

    CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::nn::layer
