#include "ops.h"

namespace sorei::cuda {

constexpr int BLOCK_SIZE = 256;

__global__ void add_kernel(const float* a, const float* b, float* out, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        out[idx] = a[idx] + b[idx];
}

void add(
    const matrix::DeviceMatrix<float>& a,
    const matrix::DeviceMatrix<float>& b,
    matrix::DeviceMatrix<float>& out
) {
    SOREI_CHECK(a.size() == b.size());
    SOREI_CHECK(a.size() == out.size());

    SOREI_CHECK(a.data());
    SOREI_CHECK(b.data());
    SOREI_CHECK(out.data());

    const int grids = ceil_div(a.size(), BLOCK_SIZE);
    add_kernel<<<grids, BLOCK_SIZE>>>(a.data(), b.data(), out.data(), a.size());
    SOREI_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void set_kernel(float* a, const float v, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        a[idx] = v;
}

void set(matrix::DeviceMatrix<float>& a, const float v) {
    SOREI_CHECK(a.data());

    const int grids = ceil_div(a.size(), BLOCK_SIZE);
    set_kernel<<<grids, BLOCK_SIZE>>>(a.data(), v, a.size());
    SOREI_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::cuda
