#include "concat.h"

namespace sorei::nn {

constexpr int BLOCK_SIZE = 256;

template <ConcatAxis Axis, bool Overwrite>
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
    if constexpr (Overwrite)
        in_g[row + col * in_r] = out_g[out_row + out_col * out_r];
    else
        in_g[row + col * in_r] += out_g[out_row + out_col * out_r];
}

void Concat::backward() {
    auto& out_g = grad();
    SOREI_CHECK(out_g.data());

    int offset = 0;
    for (const auto& input : inputs_) {
        auto& in_g = input->grad();
        SOREI_CHECK(in_g.data());

        const int blocks = cuda::ceil_div(in_g.size(), BLOCK_SIZE);

        if (axis_ == ConcatAxis::Rows) {
            SOREI_CHECK(in_g.cols() == out_g.cols());
            SOREI_CHECK(offset + in_g.rows() <= out_g.rows());

            if (input->consume_grad_write()) {
                concat_bwd_kernel<ConcatAxis::Rows, true><<<blocks, BLOCK_SIZE>>>(
                    in_g.data(), out_g.data(), in_g.rows(), in_g.cols(), out_g.rows(), offset
                );
            } else {
                concat_bwd_kernel<ConcatAxis::Rows, false><<<blocks, BLOCK_SIZE>>>(
                    in_g.data(), out_g.data(), in_g.rows(), in_g.cols(), out_g.rows(), offset
                );
            }
        } else {
            SOREI_CHECK(in_g.rows() == out_g.rows());
            SOREI_CHECK(offset + in_g.cols() <= out_g.cols());

            if (input->consume_grad_write()) {
                concat_bwd_kernel<ConcatAxis::Cols, true><<<blocks, BLOCK_SIZE>>>(
                    in_g.data(), out_g.data(), in_g.rows(), in_g.cols(), out_g.rows(), offset
                );
            } else {
                concat_bwd_kernel<ConcatAxis::Cols, false><<<blocks, BLOCK_SIZE>>>(
                    in_g.data(), out_g.data(), in_g.rows(), in_g.cols(), out_g.rows(), offset
                );
            }
        }

        SOREI_CUDA_KERNEL_LAUNCH_CHECK();

        offset += (axis_ == ConcatAxis::Rows) ? in_g.rows() : in_g.cols();
    }
}

} // namespace sorei::nn
