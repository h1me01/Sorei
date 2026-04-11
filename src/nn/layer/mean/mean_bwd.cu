#include "mean.h"

namespace nn::layer {

constexpr int BLOCK_SIZE = 256;

__global__ void mean_bwd_kernel(float* in_g, const float* out_g, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        in_g[idx] += out_g[0] / size;
}

void Mean::backward() {
    auto& in_g = input_->grad();
    auto& out_g = grad();

    if (!input_->requires_grad())
        return;

    CHECK(out_g.size() == 1);
    CHECK(out_g.data());

    const int size = in_g.size();
    const int grid_size = cuda::ceil_div(size, BLOCK_SIZE);
    mean_bwd_kernel<<<grid_size, BLOCK_SIZE>>>(in_g.data(), out_g.data(), size);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace nn::layer
