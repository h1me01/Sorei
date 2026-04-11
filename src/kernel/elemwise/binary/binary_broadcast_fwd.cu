#include "binary.h"

namespace kernel {

constexpr int BLOCK_SIZE = 1024;

template <typename Op>
__global__ void binary_broadcast_forward_kernel(
    const float* full, const float* broadcast, float* out, const int out_r, const int size, Op op
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        out[idx] = op.forward(full[idx], broadcast[idx % out_r]);
}

void elemwise_binary_broadcast_forward(
    const tensor::GPUMatrix<float>& a,
    const tensor::GPUMatrix<float>& b,
    tensor::GPUMatrix<float>& c,
    const BinaryOp& op
) {
    const bool repeat_a = (a.cols() == 1);
    const auto& full = repeat_a ? b : a;
    const auto& broadcast = repeat_a ? a : b;

    CHECK(full.shape() == c.shape());
    CHECK(full.rows() == broadcast.rows());
    CHECK(broadcast.cols() == 1);

    CHECK(a.data());
    CHECK(b.data());
    CHECK(c.data());

    const int grid = cuda::ceil_div(c.size(), BLOCK_SIZE);

    std::visit(
        [&](auto&& o) {
            binary_broadcast_forward_kernel<<<grid, BLOCK_SIZE>>>(
                full.data(), broadcast.data(), c.data(), c.rows(), c.size(), o
            );
        },
        op
    );

    CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace kernel
