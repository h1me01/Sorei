#include "concat.h"

namespace nn::layer {

constexpr int BLOCK_SIZE = 256;

template <ConcatAxis Axis>
__global__ void concat_bwd_kernel(
    float* in_g,
    const float* out_g,
    const int in_r,
    const int in_c,
    const int out_r,
    const int offset
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= in_r * in_c)
        return;

    const int col = idx / in_r;
    const int row = idx % in_r;

    const int out_row = (Axis == ConcatAxis::Rows) ? row + offset : row;
    const int out_col = (Axis == ConcatAxis::Rows) ? col : col + offset;
    in_g[row + col * in_r] += out_g[out_row + out_col * out_r];
}

void Concat::backward() {
    auto& out_g = grad();
    CHECK(out_g.data());

    int offset = 0;
    for (const auto& input : inputs_) {
        auto& in_g = input->grad();
        CHECK(in_g.data());

        const int blocks = cuda::ceil_div(in_g.size(), BLOCK_SIZE);

        if (axis_ == ConcatAxis::Rows) {
            CHECK(in_g.cols() == out_g.cols());
            CHECK(offset + in_g.rows() <= out_g.rows());

            concat_bwd_kernel<ConcatAxis::Rows><<<blocks, BLOCK_SIZE>>>(
                in_g.data(), out_g.data(), in_g.rows(), in_g.cols(), out_g.rows(), offset
            );
        } else {
            CHECK(in_g.rows() == out_g.rows());
            CHECK(offset + in_g.cols() <= out_g.cols());

            concat_bwd_kernel<ConcatAxis::Cols><<<blocks, BLOCK_SIZE>>>(
                in_g.data(), out_g.data(), in_g.rows(), in_g.cols(), out_g.rows(), offset
            );
        }

        CUDA_KERNEL_LAUNCH_CHECK();

        offset += (axis_ == ConcatAxis::Rows) ? in_g.rows() : in_g.cols();
    }
}

} // namespace nn::layer
