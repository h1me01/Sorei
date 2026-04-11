#include "concat.h"

namespace nn::layer {

constexpr int BLOCK_SIZE = 256;

template <ConcatAxis Axis>
__global__ void concat_fwd_kernel(
    const float* in, float* out, const int in_r, const int in_c, const int out_r, const int offset
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= in_r * in_c)
        return;

    const int col = idx / in_r;
    const int row = idx % in_r;

    const int out_row = (Axis == ConcatAxis::Rows) ? row + offset : row;
    const int out_col = (Axis == ConcatAxis::Rows) ? col : col + offset;
    out[out_row + out_col * out_r] = in[row + col * in_r];
}

void Concat::forward() {
    auto& out = data();
    CHECK(out.data());

    int offset = 0;
    for (const auto& input : inputs_) {
        auto& in = input->data();
        CHECK(in.data());

        const int blocks = cuda::ceil_div(in.size(), BLOCK_SIZE);

        if (axis_ == ConcatAxis::Rows) {
            CHECK(in.cols() == out.cols());
            CHECK(offset + in.rows() <= out.rows());

            concat_fwd_kernel<ConcatAxis::Rows><<<blocks, BLOCK_SIZE>>>(
                in.data(), out.data(), in.rows(), in.cols(), out.rows(), offset
            );
        } else {
            CHECK(in.rows() == out.rows());
            CHECK(offset + in.cols() <= out.cols());

            concat_fwd_kernel<ConcatAxis::Cols><<<blocks, BLOCK_SIZE>>>(
                in.data(), out.data(), in.rows(), in.cols(), out.rows(), offset
            );
        }

        CUDA_KERNEL_LAUNCH_CHECK();

        offset += (axis_ == ConcatAxis::Rows) ? in.rows() : in.cols();
    }
}

} // namespace nn::layer
