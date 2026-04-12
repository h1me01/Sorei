#include "mean.h"

namespace sorei::nn::layer {

constexpr int BLOCK_SIZE = 256;

__global__ void mean_fwd_kernel(const float* in, float* out, const int size) {
    __shared__ float shared[BLOCK_SIZE];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    shared[threadIdx.x] = (idx < size) ? in[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(out, shared[0] / size);
}

void Mean::forward() {
    auto& in = input_->data();
    auto& out = data();

    SOREI_CHECK(out.size() == 1);

    SOREI_CHECK(in.data());
    SOREI_CHECK(out.data());

    out.clear();

    const int grid_size = cuda::ceil_div(in.size(), BLOCK_SIZE);
    mean_fwd_kernel<<<grid_size, BLOCK_SIZE>>>(in.data(), out.data(), in.size());

    SOREI_CUDA_CHECK(cudaGetLastError());
}

} // namespace sorei::nn::layer
