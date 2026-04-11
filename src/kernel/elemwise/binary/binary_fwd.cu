#include "binary.h"

namespace sorei::kernel {

constexpr int BLOCK_SIZE = 1024;

template <typename Op>
__global__ void
binary_forward_kernel(const float* a, const float* b, float* c, const int n, Op op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;

    if (vec_idx >= n)
        return;

    if (vec_idx + 4 <= n) {
        float4 a4 = as_vec<const float4>(a)[idx];
        float4 b4 = as_vec<const float4>(b)[idx];
        float4 c4;
        c4.x = op.forward(a4.x, b4.x);
        c4.y = op.forward(a4.y, b4.y);
        c4.z = op.forward(a4.z, b4.z);
        c4.w = op.forward(a4.w, b4.w);
        as_vec<float4>(c)[idx] = c4;
    } else {
        for (int i = vec_idx; i < n; i++)
            c[i] = op.forward(a[i], b[i]);
    }
}

void elemwise_binary_forward(
    const tensor::GPUMatrix<float>& a,
    const tensor::GPUMatrix<float>& b,
    tensor::GPUMatrix<float>& c,
    const BinaryOp& op
) {
    CHECK(a.size() == b.size());
    CHECK(a.size() == c.size());

    CHECK(a.data());
    CHECK(b.data());
    CHECK(c.data());

    const int grid = cuda::ceil_div(a.size(), 4 * BLOCK_SIZE);

    std::visit(
        [&](auto&& o) {
            binary_forward_kernel<<<grid, BLOCK_SIZE>>>(a.data(), b.data(), c.data(), a.size(), o);
        },
        op
    );

    CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::kernel
