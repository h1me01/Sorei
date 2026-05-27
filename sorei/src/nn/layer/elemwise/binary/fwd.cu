#include "binary.h"

namespace sorei::nn {

constexpr int BLOCK_SIZE = 1024;

template <typename Op>
__global__ void
binary_forward_kernel(const float* a, const float* b, float* c, const int n, Op op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = op.forward(a[idx], b[idx]);
}

void ElemwiseBinary::forward() {
    auto& a = input1_->data();
    auto& b = input2_->data();
    auto& c = data();

    SOREI_CHECK(a.size() == b.size());
    SOREI_CHECK(a.size() == c.size());

    SOREI_CHECK(a.data());
    SOREI_CHECK(b.data());
    SOREI_CHECK(c.data());

    const int grid = cuda::ceil_div(a.size(), BLOCK_SIZE);

    std::visit(
        [&](auto&& o) {
            binary_forward_kernel<<<grid, BLOCK_SIZE>>>(a.data(), b.data(), c.data(), a.size(), o);
        },
        op_
    );

    SOREI_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::nn
