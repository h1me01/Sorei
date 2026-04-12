#pragma once

#include <cuda/cmath>
#include <cuda_runtime.h>

#include "../tensor/include.h"

#define SOREI_DEFINE_T4_OP(name, op)                                                               \
    template <typename T>                                                                          \
    __device__ __forceinline__ T name(T a, T b) {                                                  \
        a.x op## = b.x;                                                                            \
        a.y op## = b.y;                                                                            \
        a.z op## = b.z;                                                                            \
        a.w op## = b.w;                                                                            \
        return a;                                                                                  \
    }

namespace sorei::cuda {

__device__ __forceinline__ float clamp(float x, float min_val, float max_val) {
    return max(min_val, min(x, max_val));
}

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

SOREI_DEFINE_T4_OP(add_t4, +)
SOREI_DEFINE_T4_OP(sub_t4, -)
SOREI_DEFINE_T4_OP(mul_t4, *)
SOREI_DEFINE_T4_OP(div_t4, /)

} // namespace sorei::cuda
