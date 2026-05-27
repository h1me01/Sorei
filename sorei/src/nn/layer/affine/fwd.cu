#include "affine.h"

namespace sorei::nn {

constexpr int BLOCK_SIZE = 1024;

constexpr float alpha = 1.0f;
constexpr float beta = 0.0f;

__global__ void bias_add_kernel(const float* bias, float* out, const int rows, const int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
        out[idx] += bias[idx % rows];
}

void Affine::forward() {
    auto& weight = weight_->data();
    auto& bias = bias_->data();
    auto& in = input_->data();
    auto& out = data();

    SOREI_CHECK(weight.rows() == out.rows());
    SOREI_CHECK(weight.cols() == in.rows());
    SOREI_CHECK(bias.cols() == 1);
    SOREI_CHECK(bias.rows() == out.rows());
    SOREI_CHECK(in.cols() == out.cols());

    SOREI_CHECK(weight.data());
    SOREI_CHECK(bias.data());
    SOREI_CHECK(in.data());
    SOREI_CHECK(out.data());

    cublas::sgemm(false, false, alpha, weight, in, beta, out);

    const int grid = cuda::ceil_div(out.size(), BLOCK_SIZE);
    bias_add_kernel<<<grid, BLOCK_SIZE>>>(bias.data(), out.data(), out.rows(), out.cols());

    SOREI_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::nn
