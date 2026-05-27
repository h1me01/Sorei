#include "affine.h"

namespace sorei::nn {

constexpr int BLOCK_SIZE = 1024;

constexpr float alpha = 1.0f;

__global__ void
bias_add_backward_kernel(float* bias_g, const float* out_g, const int rows, const int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols)
        return;

    const float val = out_g[idx];
    if (val != 0.0f)
        atomicAdd(&bias_g[idx % rows], val);
}

void Affine::backward() {
    const bool ow_weight_g = weight_->consume_grad_write();
    const bool ow_bias_g = bias_->consume_grad_write();
    const bool ow_in_g = input_->consume_grad_write();

    auto& weight = weight_->data();
    auto& weight_g = weight_->grad();
    auto& bias_g = bias_->grad();
    auto& in = input_->data();
    auto& in_g = input_->grad();
    auto& out_g = grad();

    SOREI_CHECK(weight.rows() == out_g.rows());
    SOREI_CHECK(bias_g.cols() == 1);
    SOREI_CHECK(bias_g.rows() == out_g.rows());

    SOREI_CHECK(weight.data());
    SOREI_CHECK(bias_g.data());
    SOREI_CHECK(in.data());
    SOREI_CHECK(out_g.data());

    if (ow_bias_g)
        bias_g.clear();

    const int grid = cuda::ceil_div(out_g.size(), BLOCK_SIZE);
    bias_add_backward_kernel<<<grid, BLOCK_SIZE>>>(
        bias_g.data(), out_g.data(), out_g.rows(), out_g.cols()
    );
    SOREI_CUDA_KERNEL_LAUNCH_CHECK();

    if (!weight_g.empty())
        cublas::sgemm(false, true, alpha, out_g, in, ow_weight_g ? 0.0f : 1.0f, weight_g);

    if (!in_g.empty()) {
        SOREI_CHECK(in_g.cols() == out_g.cols());
        SOREI_CHECK(in_g.rows() == weight.cols());
        cublas::sgemm(true, false, alpha, weight, out_g, ow_in_g ? 0.0f : 1.0f, in_g);
    }
}

} // namespace sorei::nn
