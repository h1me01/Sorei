#pragma once

#include "../../util.h"
#include "../common.h"

namespace kernel {

struct AddBinary {
    static constexpr std::string_view name = "Add";

    __device__ float forward(float a, float b) const { return a + b; }
    __device__ void backward(float go, float a, float b, float& ga, float& gb) const {
        ga += go;
        gb += go;
    }
};

struct SubBinary {
    static constexpr std::string_view name = "Sub";

    __device__ float forward(float a, float b) const { return a - b; }
    __device__ void backward(float go, float a, float b, float& ga, float& gb) const {
        ga += go;
        gb -= go;
    }
};

struct MulBinary {
    static constexpr std::string_view name = "Mul";

    __device__ float forward(float a, float b) const { return a * b; }
    __device__ void backward(float go, float a, float b, float& ga, float& gb) const {
        ga += go * b;
        gb += go * a;
    }
};

struct DivBinary {
    static constexpr std::string_view name = "Div";

    __device__ float forward(float a, float b) const { return a / b; }
    __device__ void backward(float go, float a, float b, float& ga, float& gb) const {
        ga += go / b;
        gb += -go * a / (b * b);
    }
};

using BinaryOp = std::variant<AddBinary, SubBinary, MulBinary, DivBinary>;

void elemwise_binary_forward(
    const data::GPUMatrix<float>& a,
    const data::GPUMatrix<float>& b,
    data::GPUMatrix<float>& c,
    const BinaryOp& op
);

void elemwise_binary_backward(
    const data::GPUMatrix<float>& a,
    data::GPUMatrix<float>& a_g,
    const data::GPUMatrix<float>& b,
    data::GPUMatrix<float>& b_g,
    const data::GPUMatrix<float>& c_g,
    const BinaryOp& op
);

void elemwise_binary_broadcast_forward(
    const data::GPUMatrix<float>& a,
    const data::GPUMatrix<float>& b,
    data::GPUMatrix<float>& c,
    const BinaryOp& op
);

void elemwise_binary_broadcast_backward(
    const data::GPUMatrix<float>& a,
    data::GPUMatrix<float>& a_g,
    const data::GPUMatrix<float>& b,
    data::GPUMatrix<float>& b_g,
    const data::GPUMatrix<float>& c_g,
    const BinaryOp& op
);

} // namespace kernel
