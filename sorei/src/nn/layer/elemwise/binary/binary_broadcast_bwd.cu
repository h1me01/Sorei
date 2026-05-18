#include "binary.h"

namespace sorei::nn {

constexpr int BLOCK_SIZE = 1024;

template <typename Op, bool GradA, bool GradB, bool OverwriteA>
__global__ void binary_broadcast_backward_kernel(
    const float* full,
    const float* broadcast,
    float* full_g,
    float* broadcast_g,
    const float* out_g,
    const int out_r,
    const int size,
    Op op
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    const int b_idx = idx % out_r;

    const float o_g = out_g[idx];
    float a_g = 0.0f;
    float b_g = 0.0f;

    op.backward(full[idx], broadcast[b_idx], a_g, b_g, o_g);

    if constexpr (GradA) {
        if constexpr (OverwriteA)
            full_g[idx] = a_g;
        else
            full_g[idx] += a_g;
    }

    if constexpr (GradB)
        if (b_g != 0.0f)
            atomicAdd(&broadcast_g[b_idx], b_g);
}

void ElemwiseBinary::broadcast_backward(
    const matrix::DeviceMatrix<float>& a,
    matrix::DeviceMatrix<float>& a_g,
    const matrix::DeviceMatrix<float>& b,
    matrix::DeviceMatrix<float>& b_g,
    const matrix::DeviceMatrix<float>& c_g,
    const Op& op,
    bool ow_a_g,
    bool ow_b_g
) {
    const bool repeat_a = (a.cols() == 1);

    const auto& full = repeat_a ? b : a;
    auto& full_g = repeat_a ? b_g : a_g;

    const auto& broadcast = repeat_a ? a : b;
    auto& broadcast_g = repeat_a ? a_g : b_g;

    const bool ow_full = repeat_a ? ow_b_g : ow_a_g;

    SOREI_CHECK(full.shape() == c_g.shape());
    SOREI_CHECK(full.rows() == broadcast.rows());
    SOREI_CHECK(broadcast.cols() == 1);

    SOREI_CHECK(full.data());
    SOREI_CHECK(broadcast.data());
    SOREI_CHECK(c_g.data());

    SOREI_CHECK(c_g.data() != full.data());
    SOREI_CHECK(c_g.data() != broadcast.data());

    const bool ow_broadcast = repeat_a ? ow_a_g : ow_b_g;
    if (ow_broadcast && !broadcast_g.empty())
        broadcast_g.clear();

    const int grid = cuda::ceil_div(c_g.size(), BLOCK_SIZE);

    std::visit(
        [&](auto&& o) {
            auto launch = [&]<bool GradF, bool GradB, bool OvF>() {
                binary_broadcast_backward_kernel<std::decay_t<decltype(o)>, GradF, GradB, OvF>
                    <<<grid, BLOCK_SIZE>>>(
                        full.data(),
                        broadcast.data(),
                        full_g.data(),
                        broadcast_g.data(),
                        c_g.data(),
                        c_g.rows(),
                        c_g.size(),
                        o
                    );
            };

            bool ha = !full_g.empty(), hb = !broadcast_g.empty();
            if (ha && hb) {
                if (ow_full)
                    launch.template operator()<true, true, true>();
                else
                    launch.template operator()<true, true, false>();
            } else if (ha) {
                if (ow_full)
                    launch.template operator()<true, false, true>();
                else
                    launch.template operator()<true, false, false>();
            } else if (hb) {
                launch.template operator()<false, true, false>();
            }
        },
        op
    );

    SOREI_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::nn
