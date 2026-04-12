#include "../sparse_affine.h"

namespace sorei::nn::layer {

constexpr int BLOCK_SIZE = 256;

template <typename Op>
__global__ void sparse_affine_fwd_vec_kernel(
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

    if (row4 >= weight_r4 || batch_idx >= batch_size)
        return;

    const int* sample_indices = indices + batch_idx * max_entries;
    for (int i = threadIdx.x; i < max_entries; i += blockDim.x)
        shared_indices[i] = sample_indices[i];
    __syncthreads();

    const float4* w4 = cuda::as_vec<const float4>(weight);

    float4 val = cuda::as_vec<const float4>(bias)[row4];
    for (int i = 0; i < max_entries; i++) {
        const int idx = shared_indices[i];
        if (idx == -1)
            break;
        val = cuda::add_t4(val, w4[idx * weight_r4 + row4]);
    }

    val.x = op.forward(val.x);
    val.y = op.forward(val.y);
    val.z = op.forward(val.z);
    val.w = op.forward(val.w);

    cuda::as_vec<float4>(out)[out_r4 * batch_idx + row4] = val;
}

template <typename Op>
__global__ void sparse_affine_fwd_kernel(
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

    if (row >= weight_r || batch_idx >= batch_size)
        return;

    const int* sample_indices = indices + batch_idx * max_entries;

    float sum = bias[row];
    for (int i = 0; i < max_entries; i++) {
        const int idx = sample_indices[i];
        if (idx == -1)
            break;
        sum += weight[idx * weight_r + row];
    }

    out[out_r * batch_idx + row] = op.forward(sum);
}

void SparseAffine::forward() {
    auto& weight = weight_->data();
    auto& bias = bias_->data();
    auto& out = effective_data();
    auto& indices = input_->data();

    SOREI_CHECK(weight.data());
    SOREI_CHECK(bias.data());
    SOREI_CHECK(out.data());
    SOREI_CHECK(indices.data());

    SOREI_CHECK(weight.rows() == bias.rows());
    SOREI_CHECK(out.cols() <= 65535);
    SOREI_CHECK(out.rows() >= weight.rows() + out_offset_);

    const int w_r = weight.rows();

    SOREI_CHECK(out.cols() <= 65535);
    SOREI_CHECK(indices.cols() == out.cols());
    SOREI_CHECK(weight.rows() == bias.rows());
    SOREI_CHECK(out.rows() >= w_r + out_offset_);

    const int max_entries = indices.rows();

    const bool use_vec = (w_r % 4 == 0);
    const int effective_rows = use_vec ? w_r / 4 : w_r;
    const int threads = min(effective_rows, BLOCK_SIZE);
    const int row_blocks = cuda::ceil_div(effective_rows, threads);

    dim3 grid(row_blocks, out.cols());

    std::visit(
        [&](auto op) {
            if (use_vec) {
                const int shared_mem_size = max_entries * sizeof(int);

                sparse_affine_fwd_vec_kernel<<<grid, dim3(threads), shared_mem_size>>>(
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
                sparse_affine_fwd_kernel<<<grid, dim3(threads)>>>(
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
