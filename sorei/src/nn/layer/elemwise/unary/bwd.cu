#include "unary.h"

namespace sorei::nn {

constexpr int BLOCK_SIZE = 1024;

template <typename Op, bool Overwrite>
__global__ void
unary_bwd_kernel(const float* in_d, float* in_g, const float* out_g, const int size, Op op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    const float val = op.backward(in_d[idx]) * out_g[idx];
    if constexpr (Overwrite)
        in_g[idx] = val;
    else
        in_g[idx] += val;
}

void ElemwiseUnary::backward(
    matrix::DeviceMatrix<float>& in,
    matrix::DeviceMatrix<float>& in_g,
    const matrix::DeviceMatrix<float>& out_g,
    const Op& op,
    bool overwrite
) {
    if (in_g.empty())
        return;

    SOREI_CHECK(in.size() == out_g.size());

    SOREI_CHECK(in.data());
    SOREI_CHECK(out_g.data());

    const int grid = cuda::ceil_div(in.size(), BLOCK_SIZE);

    std::visit(
        [&](auto&& o) {
            if (overwrite)
                unary_bwd_kernel<std::decay_t<decltype(o)>, true>
                    <<<grid, BLOCK_SIZE>>>(in.data(), in_g.data(), out_g.data(), in.size(), o);
            else
                unary_bwd_kernel<std::decay_t<decltype(o)>, false>
                    <<<grid, BLOCK_SIZE>>>(in.data(), in_g.data(), out_g.data(), in.size(), o);
        },
        op
    );

    SOREI_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::nn
