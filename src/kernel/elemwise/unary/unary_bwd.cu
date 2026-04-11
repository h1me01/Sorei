#include "unary.h"

namespace kernel {

constexpr int BLOCK_SIZE = 1024;

template <typename Op>
__global__ void
unary_bwd_kernel(const float* in_d, float* in_g, const float* out_g, const int size, Op op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;

    if (vec_idx >= size)
        return;

    if (vec_idx + 4 <= size) {
        float4 in_d4 = as_vec<const float4>(in_d)[idx];
        float4 in_g4 = as_vec<float4>(in_g)[idx];
        float4 out_g4 = as_vec<const float4>(out_g)[idx];

        in_g4.x += op.backward(in_d4.x) * out_g4.x;
        in_g4.y += op.backward(in_d4.y) * out_g4.y;
        in_g4.z += op.backward(in_d4.z) * out_g4.z;
        in_g4.w += op.backward(in_d4.w) * out_g4.w;

        as_vec<float4>(in_g)[idx] = in_g4;
    } else {
        for (int i = vec_idx; i < size; i++)
            in_g[i] += op.backward(in_d[i]) * out_g[i];
    }
}

void elemwise_unary_backward(
    tensor::GPUMatrix<float>& in,
    tensor::GPUMatrix<float>& in_g,
    const tensor::GPUMatrix<float>& out_g,
    const UnaryOp& op
) {
    if (in_g.empty())
        return;

    CHECK(in.size() == out_g.size());

    CHECK(in.data());
    CHECK(out_g.data());

    const int grid = cuda::ceil_div(in.size(), 4 * BLOCK_SIZE);

    std::visit(
        [&](auto&& o) {
            unary_bwd_kernel<<<grid, BLOCK_SIZE>>>(
                in.data(), in_g.data(), out_g.data(), in.size(), o
            );
        },
        op
    );

    CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace kernel
