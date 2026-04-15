#include "../sparse_affine.h"

namespace sorei::nn::layer {

constexpr int BLOCK_SIZE = 512;
constexpr dim3 block_size(BLOCK_SIZE, 1);

template <typename Op>
__global__ void sparse_affine_bwd_kernel(
    float* weight_g,
    float* bias_g,
    const float* out,
    const float* out_g,
    const int* features,
    const int weight_r,
    const int out_r,
    const int batch_size,
    const int max_entries,
    Op op
) {
    const int row = blockIdx.y * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.x;

    if (row >= weight_r || batch_idx >= batch_size)
        return;

    const int* sample_indices = features + batch_idx * max_entries;
    const int out_idx = batch_idx * out_r + row;

    float out_val = out[out_idx];
    if constexpr (std::is_same_v<Op, cuda::SquaredClampedReLU>)
        out_val = sqrtf(out_val);

    const float grad = out_g[out_idx] * op.backward(out_val);
    if (grad == 0.0f)
        return;

    atomicAdd(&bias_g[row], grad);

    for (int i = 0; i < max_entries; i++) {
        const int idx = sample_indices[i];
        if (idx == -1)
            break;
        atomicAdd(&weight_g[idx * weight_r + row], grad);
    }
}

void SparseAffine::backward() {
    auto& weight_g = weight_->grad();
    auto& bias_g = bias_->grad();
    auto& out = effective_data();
    auto& out_g = effective_grad();
    auto& indices = input_->data();

    SOREI_CHECK(weight_g.data());
    SOREI_CHECK(bias_g.data());
    SOREI_CHECK(out.data());
    SOREI_CHECK(out_g.data());
    SOREI_CHECK(indices.data());

    SOREI_CHECK(out_g.cols() <= 65535);
    SOREI_CHECK(indices.cols() == out_g.cols());
    SOREI_CHECK(weight_g.rows() == bias_g.rows());
    SOREI_CHECK(out_g.rows() >= weight_g.rows() + out_offset_);

    const int row_tiles = cuda::ceil_div(weight_g.rows(), BLOCK_SIZE);
    dim3 grid(out_g.cols(), row_tiles);

    std::visit(
        [&](auto op) {
            sparse_affine_bwd_kernel<<<grid, block_size>>>(
                weight_g.data(),
                bias_g.data(),
                out.data() + out_offset_,
                out_g.data() + out_offset_,
                indices.data(),
                weight_g.rows(),
                out_g.rows(),
                out_g.cols(),
                indices.rows(),
                op
            );
        },
        act_op_
    );

    SOREI_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::nn::layer
