#include "../../cuda_utils.h"

#include "adam.h"

namespace sorei::nn {

constexpr int BLOCK_SIZE = 1024;

__global__ void adam_kernel(
    float* data,
    float* moms,
    float* vels,
    const float* grads,
    const float lr,
    const float beta1,
    const float beta2,
    const float decay,
    const float min_val,
    const float max_val,
    const int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_idx = idx * 4;
    if (vec_idx >= size)
        return;

    const auto update = [&](float& val, float& mom, float& vel, float grad) {
        val *= decay;
        mom = beta1 * mom + (1.0f - beta1) * grad;
        vel = beta2 * vel + (1.0f - beta2) * grad * grad;
        val -= lr * mom / (sqrtf(vel) + 1e-8f);
        val = max(min_val, min(max_val, val));
    };

    if (vec_idx + 4 <= size) {
        float4 val = cuda::as_vec<float4>(data)[idx];
        float4 mom = cuda::as_vec<float4>(moms)[idx];
        float4 vel = cuda::as_vec<float4>(vels)[idx];
        const float4 grad = cuda::as_vec<const float4>(grads)[idx];

        update(val.x, mom.x, vel.x, grad.x);
        update(val.y, mom.y, vel.y, grad.y);
        update(val.z, mom.z, vel.z, grad.z);
        update(val.w, mom.w, vel.w, grad.w);

        cuda::as_vec<float4>(data)[idx] = val;
        cuda::as_vec<float4>(moms)[idx] = mom;
        cuda::as_vec<float4>(vels)[idx] = vel;
    } else {
        for (int i = vec_idx; i < size; i++)
            update(data[i], moms[i], vels[i], grads[i]);
    }
}

void AdamW::step(float lr) {
    for (size_t i = 0; i < params_.size(); i++) {
        auto& data = params_[i]->data();
        auto& grad = params_[i]->grad();
        auto& moms = momentum_[i];
        auto& vels = velocity_[i];

        SOREI_CHECK(moms.size() == data.size());
        SOREI_CHECK(vels.size() == data.size());

        SOREI_CHECK(data.data());
        SOREI_CHECK(grad.data());
        SOREI_CHECK(moms.data());
        SOREI_CHECK(vels.data());

        const int blocks = cuda::ceil_div(data.size(), 4 * BLOCK_SIZE);
        adam_kernel<<<blocks, BLOCK_SIZE>>>(
            data.data(),
            moms.data(),
            vels.data(),
            grad.data(),
            lr,
            beta1_,
            beta2_,
            1.0f - lr * decay_,
            params_[i]->lower_bound(),
            params_[i]->upper_bound(),
            data.size()
        );

        SOREI_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

} // namespace sorei::nn
