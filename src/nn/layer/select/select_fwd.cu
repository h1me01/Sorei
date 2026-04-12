#include "select.h"

namespace sorei::nn::layer {

constexpr int BLOCK_SIZE = 256;

__global__ void select_fwd_kernel(
    const float* in,
    float* out,
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
    out[out_offset] = in[in_offset];
}

void Select::forward() {
    auto& in = input_->data();
    auto& out = data();
    auto& indices = bucket_->data();

    SOREI_CHECK(indices.rows() == 1);
    SOREI_CHECK(in.cols() == out.cols());
    SOREI_CHECK(out.cols() == indices.cols());
    SOREI_CHECK(in.rows() % out.rows() == 0);

    SOREI_CHECK(in.data());
    SOREI_CHECK(out.data());
    SOREI_CHECK(indices.data());

    const int blocks = cuda::ceil_div(out.size(), BLOCK_SIZE);
    select_fwd_kernel<<<blocks, BLOCK_SIZE>>>(
        in.data(), out.data(), indices.data(), in.rows(), out.rows(), out.cols()
    );

    SOREI_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::nn::layer
