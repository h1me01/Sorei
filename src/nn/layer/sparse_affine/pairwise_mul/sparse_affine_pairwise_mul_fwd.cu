#include "../sparse_affine.h"

namespace sorei::nn::layer {

constexpr int BLOCK_SIZE = 256;

template <typename Op>
__global__ void sparse_affine_pairwise_mul_fwd_vec_kernel(
    const float* weight,
    const float* bias,
    const int* indices,
    float* out,
    const int weight_r4,
    const int out_r4,
    const int batch_size,
    const int max_entries,
    Op op
) {
    extern __shared__ int shared_indices[];

    const int row4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y;

    const int half4 = weight_r4 / 2;

    if (row4 >= half4 || batch_idx >= batch_size)
        return;

    const int* sample_indices = indices + batch_idx * max_entries;
    for (int i = threadIdx.x; i < max_entries; i += blockDim.x)
        shared_indices[i] = sample_indices[i];
    __syncthreads();

    const float4* w4 = kernel::as_vec<const float4>(weight);
    const float4* b4 = kernel::as_vec<const float4>(bias);

    float4 sum_a = b4[row4];
    float4 sum_b = b4[row4 + half4];

    for (int i = 0; i < max_entries; i++) {
        const int idx = shared_indices[i];
        if (idx == -1)
            break;

        const int base = idx * weight_r4 + row4;
        sum_a = kernel::add_t4(sum_a, w4[base]);
        sum_b = kernel::add_t4(sum_b, w4[base + half4]);
    }

    sum_a.x = op.forward(sum_a.x);
    sum_a.y = op.forward(sum_a.y);
    sum_a.z = op.forward(sum_a.z);
    sum_a.w = op.forward(sum_a.w);

    sum_b.x = op.forward(sum_b.x);
    sum_b.y = op.forward(sum_b.y);
    sum_b.z = op.forward(sum_b.z);
    sum_b.w = op.forward(sum_b.w);

    kernel::as_vec<float4>(out)[out_r4 * batch_idx + row4] = kernel::mul_t4(sum_a, sum_b);
}

template <typename Op>
__global__ void sparse_affine_pairwise_mul_fwd_kernel(
    const float* weight,
    const float* bias,
    const int* indices,
    float* out,
    const int weight_r,
    const int out_r,
    const int batch_size,
    const int max_entries,
    Op op
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y;
    const int half = weight_r / 2;

    if (row >= half || batch_idx >= batch_size)
        return;

    const int* sample_indices = indices + batch_idx * max_entries;

    float sum_a = bias[row];
    float sum_b = bias[row + half];

    for (int i = 0; i < max_entries; i++) {
        const int idx = sample_indices[i];
        if (idx == -1)
            break;
        const int base = idx * weight_r + row;
        sum_a += weight[base];
        sum_b += weight[base + half];
    }

    out[out_r * batch_idx + row] = op.forward(sum_a) * op.forward(sum_b);
}

void SparseAffinePairwiseMul::forward() {
    auto& weight = weight_->data();
    auto& bias = bias_->data();
    auto& out = effective_data();
    auto& indices = input_->data();

    const int w_r = weight.rows();

    SOREI_CHECK(out.cols() <= 65535);
    SOREI_CHECK(indices.cols() == out.cols());
    SOREI_CHECK(2 * out.rows() >= w_r + out_offset_);

    SOREI_CHECK(weight.data());
    SOREI_CHECK(bias.data());
    SOREI_CHECK(out.data());
    SOREI_CHECK(indices.data());

    const int max_entries = indices.rows();

    const bool use_vec = (w_r % 8 == 0);
    const int effective_rows = use_vec ? w_r / 4 / 2 : w_r / 2;
    const int threads = min(effective_rows, BLOCK_SIZE);
    const int row_blocks = cuda::ceil_div(effective_rows, threads);

    dim3 grid(row_blocks, out.cols());

    std::visit(
        [&](auto op) {
            const int shared_mem_size = max_entries * sizeof(int);

            if (use_vec) {
                sparse_affine_pairwise_mul_fwd_vec_kernel<<<grid, dim3(threads), shared_mem_size>>>(
                    weight.data(),
                    bias.data(),
                    indices.data(),
                    out.data() + out_offset_,
                    w_r / 4,
                    out.rows() / 4,
                    out.cols(),
                    max_entries,
                    op
                );
            } else {
                sparse_affine_pairwise_mul_fwd_kernel<<<grid, dim3(threads)>>>(
                    weight.data(),
                    bias.data(),
                    indices.data(),
                    out.data() + out_offset_,
                    w_r,
                    out.rows(),
                    out.cols(),
                    max_entries,
                    op
                );
            }
        },
        act_op_
    );

    SOREI_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::nn::layer
