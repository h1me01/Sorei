#include "mean.h"

namespace sorei::nn {

constexpr int BLOCK_SIZE = 256;

template <bool Overwrite>
__global__ void mean_bwd_kernel(float* in_g, const float* out_g, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    const float val = out_g[0] / size;
    if constexpr (Overwrite)
        in_g[idx] = val;
    else
        in_g[idx] += val;
}

void Mean::backward() {
    if (!input_->requires_grad())
        return;

    auto& in_g = input_->grad();
    auto& out_g = grad();

    SOREI_CHECK(out_g.size() == 1);
    SOREI_CHECK(out_g.data());

    const int grid_size = cuda::ceil_div(in_g.size(), BLOCK_SIZE);

    if (input_->consume_grad_write())
        mean_bwd_kernel<true><<<grid_size, BLOCK_SIZE>>>(in_g.data(), out_g.data(), in_g.size());
    else
        mean_bwd_kernel<false><<<grid_size, BLOCK_SIZE>>>(in_g.data(), out_g.data(), in_g.size());

    SOREI_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::nn
