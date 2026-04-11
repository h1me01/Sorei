#pragma once

#include "../../util.h"
#include "../common.h"

namespace kernel {

// Basic arithmetic

struct Identity {
    static constexpr std::string_view name = "Identity";

    __device__ float forward(float x) const { return x; }
    __device__ float backward(float x) const { return 1.0f; }
    __device__ float backward_from_output(float y) const { return backward(y); }
};

struct AddScaleUnary {
    static constexpr std::string_view name = "AddScale";

    float scale = 1.0f;
    float bias = 0.0f;
    __device__ float forward(float x) const { return x * scale + bias; }
    __device__ float backward(float x) const { return scale; }
};

struct DivLeftUnary {
    static constexpr std::string_view name = "DivLeftScalar";

    float scalar = 1.0f;
    __device__ float forward(float x) const { return scalar / x; }
    __device__ float backward(float x) const { return -scalar / (x * x); }
};

// Unary operations

struct Clamp {
    static constexpr std::string_view name = "Clamp";

    float min_val = 0.0f;
    float max_val = 1.0f;
    __device__ float forward(float x) const { return clamp(x, min_val, max_val); }
    __device__ float backward(float x) const { return (x > min_val && x < max_val) ? 1.0f : 0.0f; }
};

struct Abs {
    static constexpr std::string_view name = "Abs";

    __device__ float forward(float x) const { return abs(x); }
    __device__ float backward(float x) const {
        return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
    }
};

struct PowInt {
    static constexpr std::string_view name = "Pow";

    int exponent;

    __device__ float forward(float x) const {
        if (exponent == 0)
            return 1.0f;

        int exp = exponent < 0 ? -exponent : exponent;
        float result = 1.0f;
        float base = x;

        while (exp > 0) {
            if (exp & 1)
                result *= base;
            base *= base;
            exp >>= 1;
        }

        return exponent < 0 ? 1.0f / result : result;
    }

    __device__ float backward(float x) const {
        if (exponent == 0)
            return 0.0f;

        if (x == 0.0f) {
            if (exponent == 1)
                return 1.0f;
            if (exponent == -1)
                return -INFINITY;
            if (exponent > 1)
                return 0.0f;
            return INFINITY;
        }

        return static_cast<float>(exponent) * PowInt{exponent - 1}.forward(x);
    }
};

struct PowFloat {
    static constexpr std::string_view name = "Pow";

    float exponent;

    __device__ float safe_pow(float x, float exp) const {
        if (exp == 0.0f)
            return 1.0f;
        if (x == 0.0f)
            return exp > 0.0f ? 0.0f : INFINITY;
        if (x > 0.0f)
            return powf(x, exp);

        float ei = roundf(exp);
        if (fabsf(exp - ei) > 1e-6f)
            return NAN;

        float r = powf(-x, exp);
        return (static_cast<int>(ei) & 1) ? -r : r;
    }

    __device__ float forward(float x) const { return safe_pow(x, exponent); }

    __device__ float backward(float x) const {
        if (exponent == 0.0f)
            return 0.0f;
        if (x == 0.0f) {
            if (exponent > 1.0f)
                return 0.0f;
            if (exponent == 1.0f)
                return 1.0f;
            return INFINITY;
        }
        return exponent * safe_pow(x, exponent - 1.0f);
    }
};

// Activation functions

struct ReLU {
    static constexpr std::string_view name = "ReLU";

    __device__ float forward(float x) const { return max(0.0f, x); }
    __device__ float backward(float x) const { return (x > 0.0f) ? 1.0f : 0.0f; }
    __device__ float backward_from_output(float y) const { return backward(y); }
};

// ReLU clamped to [0, 1]
struct ClampedReLU {
    static constexpr std::string_view name = "ClampedReLU";

    __device__ float forward(float x) const { return clamp(x, 0.0f, 1.0f); }
    __device__ float backward(float x) const { return (x > 0.0f && x < 1.0f) ? 1.0f : 0.0f; }
    __device__ float backward_from_output(float y) const { return backward(y); }
};

// ReLU clamped to [0, 1] and squared
struct SquaredClampedReLU {
    static constexpr std::string_view name = "SquaredClampedReLU";

    __device__ float forward(float x) const {
        x = clamp(x, 0.0f, 1.0f);
        return x * x;
    }
    __device__ float backward(float x) const { return (x > 0.0f && x < 1.0f) ? 2.0f * x : 0.0f; }
    __device__ float backward_from_output(float y) const { return backward(sqrtf(y)); }
};

struct Sigmoid {
    static constexpr std::string_view name = "Sigmoid";

    __device__ float forward(float x) const { return 1.0f / (1.0f + expf(-x)); }
    __device__ float backward(float x) const {
        x = forward(x);
        return x * (1.0f - x);
    }
    __device__ float backward_from_output(float y) const { return y * (1.0f - y); }
};

// ElemwiseUnary

using UnaryOp = std::variant<
    Identity,
    AddScaleUnary,
    DivLeftUnary,
    Clamp,
    Abs,
    PowInt,
    PowFloat,
    ReLU,
    ClampedReLU,
    SquaredClampedReLU,
    Sigmoid>;

using ActOp = std::variant<Identity, ReLU, ClampedReLU, SquaredClampedReLU, Sigmoid>;

void elemwise_unary_forward(
    const tensor::GPUMatrix<float>& in, tensor::GPUMatrix<float>& out, const UnaryOp& op
);

void elemwise_unary_backward(
    tensor::GPUMatrix<float>& in,
    tensor::GPUMatrix<float>& in_g,
    const tensor::GPUMatrix<float>& out_g,
    const UnaryOp& op
);

inline std::optional<kernel::ActOp> as_activation(const kernel::UnaryOp& op) {
    if (std::holds_alternative<kernel::ReLU>(op))
        return kernel::ActOp{std::get<kernel::ReLU>(op)};
    if (std::holds_alternative<kernel::ClampedReLU>(op))
        return kernel::ActOp{std::get<kernel::ClampedReLU>(op)};
    if (std::holds_alternative<kernel::SquaredClampedReLU>(op))
        return kernel::ActOp{std::get<kernel::SquaredClampedReLU>(op)};
    if (std::holds_alternative<kernel::Sigmoid>(op))
        return kernel::ActOp{std::get<kernel::Sigmoid>(op)};
    return std::nullopt;
}

} // namespace kernel
