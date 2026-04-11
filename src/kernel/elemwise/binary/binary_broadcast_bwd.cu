#include "binary.h"

namespace kernel {

constexpr int BLOCK_SIZE = 1024;

template <typename Op, bool GradF, bool GradB>
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

    const float go = out_g[idx];
    float fg = 0.0f;
    float bg = 0.0f;

    op.backward(go, full[idx], broadcast[b_idx], fg, bg);

    if constexpr (GradF)
        full_g[idx] += fg;
    if constexpr (GradB)
        if (bg != 0.0f)
            atomicAdd(&broadcast_g[b_idx], bg);
}

void elemwise_binary_broadcast_backward(
    const tensor::GPUMatrix<float>& a,
    tensor::GPUMatrix<float>& a_g,
    const tensor::GPUMatrix<float>& b,
    tensor::GPUMatrix<float>& b_g,
    const tensor::GPUMatrix<float>& c_g,
    const BinaryOp& op
) {
    const bool repeat_a = (a.cols() == 1);

    const auto& full = repeat_a ? b : a;
    auto& full_g = repeat_a ? b_g : a_g;

    const auto& broadcast = repeat_a ? a : b;
    auto& broadcast_g = repeat_a ? a_g : b_g;

    CHECK(full.shape() == c_g.shape());
    CHECK(full.rows() == broadcast.rows());
    CHECK(broadcast.cols() == 1);

    CHECK(full.data());
    CHECK(broadcast.data());
    CHECK(c_g.data());

    CHECK(c_g.data() != full.data());
    CHECK(c_g.data() != broadcast.data());

    const int grid = cuda::ceil_div(c_g.size(), BLOCK_SIZE);

    std::visit(
        [&](auto&& o) {
            auto launch = [&]<bool GradF, bool GradB>() {
                binary_broadcast_backward_kernel<decltype(o), GradF, GradB><<<grid, BLOCK_SIZE>>>(
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
            if (ha && hb)
                launch.template operator()<true, true>();
            else if (ha)
                launch.template operator()<true, false>();
            else if (hb)
                launch.template operator()<false, true>();
        },
        op
    );

    CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace kernel
