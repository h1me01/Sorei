#include "../sparse_affine.h"

namespace sorei::nn::layer {

constexpr int BLOCK_SIZE = 512;
constexpr dim3 block_size(BLOCK_SIZE, 1);

template <typename Op>
__global__ void sparse_affine_pairwise_mul_bwd_kernel(
    float* weight_g,
    float* bias_g,
    const float* weight,
    const float* bias,
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

    const int half = weight_r / 2;

    extern __shared__ int shared_features[];
    for (int i = threadIdx.x; i < max_entries; i += blockDim.x)
        shared_features[i] = features[batch_idx * max_entries + i];
    __syncthreads();

    if (row >= half || batch_idx >= batch_size)
        return;

    const float grad = out_g[batch_idx * out_r + row];
    if (grad == 0.0f)
        return;

    float sum_a = bias[row];
    float sum_b = bias[row + half];

    for (int i = 0; i < max_entries; i++) {
        const int idx = shared_features[i];
        if (idx == -1)
            break;
        const int base = weight_r * idx + row;
        sum_a += weight[base];
        sum_b += weight[base + half];
    }

    const float grad_a = grad * op.backward(sum_a) * op.forward(sum_b);
    const float grad_b = grad * op.backward(sum_b) * op.forward(sum_a);

    if (grad_a != 0.0f)
        atomicAdd(&bias_g[row], grad_a);
    if (grad_b != 0.0f)
        atomicAdd(&bias_g[row + half], grad_b);

    for (int i = 0; i < max_entries; i++) {
        const int idx = shared_features[i];
        if (idx == -1)
            break;
        const int base = weight_r * idx + row;
        if (grad_a != 0.0f)
            atomicAdd(&weight_g[base], grad_a);
        if (grad_b != 0.0f)
            atomicAdd(&weight_g[base + half], grad_b);
    }
}

void SparseAffinePairwiseMul::backward() {
    auto& weight_g = weight_->grad();
    auto& bias_g = bias_->grad();
    auto& weight = weight_->data();
    auto& bias = bias_->data();
    auto& out_g = effective_grad();
    auto& indices = input_->data();

    SOREI_CHECK(out_g.cols() <= 65535);
    SOREI_CHECK(indices.cols() == out_g.cols());
    SOREI_CHECK(weight_g.rows() == bias_g.rows());
    SOREI_CHECK(2 * out_g.rows() >= weight_g.rows() + out_offset_);

    SOREI_CHECK(weight_g.data());
    SOREI_CHECK(bias_g.data());
    SOREI_CHECK(weight.data());
    SOREI_CHECK(bias.data());
    SOREI_CHECK(out_g.data());
    SOREI_CHECK(indices.data());

    const int max_entries = indices.rows();
    const int row_tiles = cuda::ceil_div(weight_g.rows() / 2, BLOCK_SIZE);
    const int shared_mem = max_entries * sizeof(int);

    dim3 grid(out_g.cols(), row_tiles);

    std::visit(
        [&](auto op) {
            sparse_affine_pairwise_mul_bwd_kernel<<<grid, block_size, shared_mem>>>(
                weight_g.data(),
                bias_g.data(),
                weight.data(),
                bias.data(),
                out_g.data() + out_offset_,
                indices.data(),
                weight_g.rows(),
                out_g.rows(),
                out_g.cols(),
                max_entries,
                op
            );
        },
        act_op_
    );

    SOREI_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::nn::layer
