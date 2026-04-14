#include "binary.h"

namespace sorei::nn::layer {

constexpr int BLOCK_SIZE = 1024;

template <typename Op>
__global__ void
binary_forward_kernel(const float* a, const float* b, float* c, const int n, Op op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;

    if (vec_idx >= n)
        return;

    if (vec_idx + 4 <= n) {
        float4 a4 = cuda::as_vec<const float4>(a)[idx];
        float4 b4 = cuda::as_vec<const float4>(b)[idx];
        float4 c4;
        c4.x = op.forward(a4.x, b4.x);
        c4.y = op.forward(a4.y, b4.y);
        c4.z = op.forward(a4.z, b4.z);
        c4.w = op.forward(a4.w, b4.w);
        cuda::as_vec<float4>(c)[idx] = c4;
    } else {
        for (int i = vec_idx; i < n; i++)
            c[i] = op.forward(a[i], b[i]);
    }
}

void ElemwiseBinary::forward(
    const tensor::DeviceMatrix<float>& a,
    const tensor::DeviceMatrix<float>& b,
    tensor::DeviceMatrix<float>& c,
    const Op& op
) {
    SOREI_CHECK(a.size() == b.size());
    SOREI_CHECK(a.size() == c.size());

    SOREI_CHECK(a.data());
    SOREI_CHECK(b.data());
    SOREI_CHECK(c.data());

    const int grid = cuda::ceil_div(a.size(), 4 * BLOCK_SIZE);

    std::visit(
        [&](auto&& o) {
            binary_forward_kernel<<<grid, BLOCK_SIZE>>>(a.data(), b.data(), c.data(), a.size(), o);
        },
        op
    );

    SOREI_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::nn::layer
