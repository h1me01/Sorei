#include "set.h"

namespace kernel {

constexpr int BLOCK_SIZE = 256;

__global__ void set_kernel(float* data, const float val, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        data[idx] = val;
}

void set(tensor::GPUMatrix<float>& data, const float val) {
    CHECK(data.data());

    const int grid = cuda::ceil_div(data.size(), BLOCK_SIZE);
    set_kernel<<<grid, BLOCK_SIZE>>>(data.data(), val, data.size());

    CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace kernel
