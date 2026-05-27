#include "unary.h"

namespace sorei::nn {

constexpr int BLOCK_SIZE = 1024;

template <typename Op>
__global__ void unary_fwd_kernel(const float* in, float* out, const int size, Op op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        out[idx] = op.forward(in[idx]);
}

void ElemwiseUnary::forward() {
    auto& in = input_->data();
    auto& out = data();

    SOREI_CHECK(in.size() == out.size());

    SOREI_CHECK(in.data());
    SOREI_CHECK(out.data());

    const int grid = cuda::ceil_div(in.size(), BLOCK_SIZE);

    std::visit(
        [&](auto&& o) {
            unary_fwd_kernel<<<grid, BLOCK_SIZE>>>(in.data(), out.data(), in.size(), o);
        },
        op_
    );

    SOREI_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::nn
