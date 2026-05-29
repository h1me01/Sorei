#include "binary.h"

namespace sorei::nn {

constexpr int BLOCK_SIZE = 1024;

template <typename Op, bool GradA, bool GradB, bool OverwriteA, bool OverwriteB>
__global__ void binary_bwd_kernel(
    const float* a, const float* b, float* a_g, float* b_g, const float* c_g, const int size, Op op
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    float ag = 0.0f;
    float bg = 0.0f;

    op.backward(a[idx], b[idx], ag, bg, c_g[idx]);

    if constexpr (GradA) {
        if constexpr (OverwriteA)
            a_g[idx] = ag;
        else
            a_g[idx] += ag;
    }

    if constexpr (GradB) {
        if constexpr (OverwriteB)
            b_g[idx] = bg;
        else
            b_g[idx] += bg;
    }
}

void ElemwiseBinary::backward() {
    const bool ow_a_g = input1_->consume_grad_write();
    const bool ow_b_g = input2_->consume_grad_write();

    auto& a = input1_->data();
    auto& b = input2_->data();
    auto& a_g = input1_->grad();
    auto& b_g = input2_->grad();
    auto& c_g = grad();

    SOREI_CHECK(a.size() == b.size());
    SOREI_CHECK(a.size() == c_g.size());

    SOREI_CHECK(a.data());
    SOREI_CHECK(b.data());
    SOREI_CHECK(c_g.data());

    SOREI_CHECK(c_g.data() != a.data());
    SOREI_CHECK(c_g.data() != b.data());

    const int grid = cuda::ceil_div(a.size(), BLOCK_SIZE);

    std::visit(
        [&](auto&& o) {
            auto launch = [&]<bool GradA, bool GradB, bool OvA, bool OvB>() {
                binary_bwd_kernel<std::decay_t<decltype(o)>, GradA, GradB, OvA, OvB>
                    <<<grid, BLOCK_SIZE>>>(
                        a.data(), b.data(), a_g.data(), b_g.data(), c_g.data(), a.size(), o
                    );
            };

            const bool ha = !a_g.empty(), hb = !b_g.empty();
            if (ha && hb) {
                if (ow_a_g && ow_b_g)
                    launch.template operator()<true, true, true, true>();
                else if (ow_a_g)
                    launch.template operator()<true, true, true, false>();
                else if (ow_b_g)
                    launch.template operator()<true, true, false, true>();
                else
                    launch.template operator()<true, true, false, false>();
            } else if (ha) {
                if (ow_a_g)
                    launch.template operator()<true, false, true, false>();
                else
                    launch.template operator()<true, false, false, false>();
            } else if (hb) {
                if (ow_b_g)
                    launch.template operator()<false, true, false, true>();
                else
                    launch.template operator()<false, true, false, false>();
            }
        },
        op_
    );

    SOREI_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::nn
