#include "unary.h"

namespace kernel {

constexpr int BLOCK_SIZE = 1024;

template <typename Op>
__global__ void unary_fwd_kernel(const float* in, float* out, const int size, Op op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;

    if (vec_idx >= size)
        return;

    if (vec_idx + 4 <= size) {
        float4 in4 = as_vec<const float4>(in)[idx];
        float4 out4;
        out4.x = op.forward(in4.x);
        out4.y = op.forward(in4.y);
        out4.z = op.forward(in4.z);
        out4.w = op.forward(in4.w);
        as_vec<float4>(out)[idx] = out4;
    } else {
        for (int i = vec_idx; i < size; i++)
            out[i] = op.forward(in[i]);
    }
}

void elemwise_unary_forward(
    const tensor::GPUMatrix<float>& in, tensor::GPUMatrix<float>& out, const UnaryOp& op
) {
    CHECK(in.size() == out.size());

    CHECK(in.data());
    CHECK(out.data());

    const int grid = cuda::ceil_div(in.size(), 4 * BLOCK_SIZE);

    std::visit(
        [&](auto&& o) {
            unary_fwd_kernel<<<grid, BLOCK_SIZE>>>(in.data(), out.data(), in.size(), o);
        },
        op
    );

    CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace kernel
