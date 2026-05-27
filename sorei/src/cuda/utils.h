#pragma once

#include <cuda/cmath>
#include <cuda_runtime.h>

#define SOREI_CUDA_CHECK(expr)                                                                     \
    do {                                                                                           \
        cudaError_t result = (expr);                                                               \
        if (result != cudaSuccess) {                                                               \
            printf("CUDA error: error when calling %s\n", #expr);                                  \
            printf("    file: %s\n", __FILE__);                                                    \
            printf("    line: %d\n", __LINE__);                                                    \
            printf("    error: %s\n", cudaGetErrorString(result));                                 \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

#define SOREI_CUDA_KERNEL_LAUNCH_CHECK() SOREI_CUDA_CHECK(cudaGetLastError())

namespace sorei::cuda {

template <typename T, typename U>
__device__ __forceinline__ const T* as_vec(const U* ptr) {
    return reinterpret_cast<const T*>(ptr);
}

template <typename T, typename U>
__device__ __forceinline__ T* as_vec(U* ptr) {
    return reinterpret_cast<T*>(ptr);
}

template <typename T>
__host__ __device__ constexpr T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
__device__ __forceinline__ T add_t4(T a, T b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

template <typename T>
__device__ __forceinline__ T mul_t4(T a, T b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
    return a;
}

} // namespace sorei::cuda
