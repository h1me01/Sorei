#include "binary.h"

namespace sorei::nn::layer {

constexpr int BLOCK_SIZE = 1024;

template <typename Op, bool GradF, bool GradB>
__global__ void binary_backward_kernel(
    const float* a, const float* b, float* a_g, float* b_g, const float* c_g, const int size, Op op
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;

    if (vec_idx >= size)
        return;

    if (vec_idx + 4 <= size) {
        float4 c_g4 = cuda::as_vec<const float4>(c_g)[idx];
        float4 a4 = cuda::as_vec<const float4>(a)[idx];
        float4 b4 = cuda::as_vec<const float4>(b)[idx];

        float4 a_g4 = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 b_g4 = {0.0f, 0.0f, 0.0f, 0.0f};

        op.backward(c_g4.x, a4.x, b4.x, a_g4.x, b_g4.x);
        op.backward(c_g4.y, a4.y, b4.y, a_g4.y, b_g4.y);
        op.backward(c_g4.z, a4.z, b4.z, a_g4.z, b_g4.z);
        op.backward(c_g4.w, a4.w, b4.w, a_g4.w, b_g4.w);

        if constexpr (GradF) {
            float4& a_g_ref = cuda::as_vec<float4>(a_g)[idx];
            a_g_ref = cuda::add_t4(a_g_ref, a_g4);
        }

        if constexpr (GradB) {
            float4& b_g_ref = cuda::as_vec<float4>(b_g)[idx];
            b_g_ref = cuda::add_t4(b_g_ref, b_g4);
        }
    } else {
        for (int i = vec_idx; i < size; i++) {
            float ag = 0.0f;
            float bg = 0.0f;

            op.backward(c_g[i], a[i], b[i], ag, bg);

            if constexpr (GradF)
                a_g[i] += ag;
            if constexpr (GradB)
                b_g[i] += bg;
        }
    }
}

void ElemwiseBinary::backward(
    const tensor::GPUMatrix<float>& a,
    tensor::GPUMatrix<float>& a_g,
    const tensor::GPUMatrix<float>& b,
    tensor::GPUMatrix<float>& b_g,
    const tensor::GPUMatrix<float>& c_g,
    const Op& op
) {
    SOREI_CHECK(a.size() == b.size());
    SOREI_CHECK(a.size() == c_g.size());

    SOREI_CHECK(a.data());
    SOREI_CHECK(b.data());
    SOREI_CHECK(c_g.data());

    SOREI_CHECK(c_g.data() != a.data());
    SOREI_CHECK(c_g.data() != b.data());

    const int grid = cuda::ceil_div(a.size(), 4 * BLOCK_SIZE);

    std::visit(
        [&](auto&& o) {
            auto launch = [&]<bool GradA, bool GradB>() {
                binary_backward_kernel<decltype(o), GradA, GradB><<<grid, BLOCK_SIZE>>>(
                    a.data(), b.data(), a_g.data(), b_g.data(), c_g.data(), a.size(), o
                );
            };

            bool ha = !a_g.empty(), hb = !b_g.empty();
            if (ha && hb)
                launch.template operator()<true, true>();
            else if (ha)
                launch.template operator()<true, false>();
            else if (hb)
                launch.template operator()<false, true>();
        },
        op
    );

    SOREI_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::nn::layer
