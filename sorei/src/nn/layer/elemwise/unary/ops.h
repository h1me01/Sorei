#pragma once

#include <string_view>

namespace sorei::nn::unary {

struct Identity {
    static constexpr std::string_view name = "Identity";

    __device__ float forward(float x) const { return x; }
    __device__ float backward(float x) const { return 1.0f; }
};

struct AddScale {
    static constexpr std::string_view name = "AddScale";

    float scale = 1.0f;
    float bias = 0.0f;
    __device__ float forward(float x) const { return x * scale + bias; }
    __device__ float backward(float x) const { return scale; }
};

struct DivLeft {
    static constexpr std::string_view name = "DivLeftScalar";

    float scalar = 1.0f;
    __device__ float forward(float x) const { return scalar / x; }
    __device__ float backward(float x) const { return -scalar / (x * x); }
};

struct Clamp {
    static constexpr std::string_view name = "Clamp";

    float min_val = 0.0f;
    float max_val = 1.0f;
    __device__ float forward(float x) const { return max(min_val, min(max_val, x)); }
    __device__ float backward(float x) const { return (x > min_val && x < max_val) ? 1.0f : 0.0f; }
};

struct Abs {
    static constexpr std::string_view name = "Abs";

    __device__ float forward(float x) const { return abs(x); }
    __device__ float backward(float x) const {
        return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
    }
};

struct ReLU {
    static constexpr std::string_view name = "ReLU";

    __device__ float forward(float x) const { return max(0.0f, x); }
    __device__ float backward(float x) const { return (x > 0.0f) ? 1.0f : 0.0f; }
};

// ReLU clamped to [0, 1]
struct ClampedReLU {
    static constexpr std::string_view name = "ClampedReLU";

    __device__ float forward(float x) const { return max(0.0f, min(1.0f, x)); }
    __device__ float backward(float x) const { return (x > 0.0f && x < 1.0f) ? 1.0f : 0.0f; }
};

// ReLU clamped to [0, 1] and squared
struct SquaredClampedReLU {
    static constexpr std::string_view name = "SquaredClampedReLU";

    __device__ float forward(float x) const {
        x = max(0.0f, min(1.0f, x));
        return x * x;
    }
    __device__ float backward(float x) const { return (x > 0.0f && x < 1.0f) ? 2.0f * x : 0.0f; }
};

struct Sigmoid {
    static constexpr std::string_view name = "Sigmoid";

    __device__ float forward(float x) const { return 1.0f / (1.0f + expf(-x)); }
    __device__ float backward(float x) const {
        x = forward(x);
        return x * (1.0f - x);
    }
};

} // namespace sorei::nn::unary
