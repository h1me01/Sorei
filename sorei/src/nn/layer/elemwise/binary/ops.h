#pragma once

#include <string_view>

namespace sorei::nn::binary {

struct Add {
    static constexpr std::string_view name = "Add";

    __device__ float forward(float a, float b) const { return a + b; }
    __device__ void backward(float a, float b, float& ag, float& bg, float go) const {
        ag += go;
        bg += go;
    }
};

struct Sub {
    static constexpr std::string_view name = "Sub";

    __device__ float forward(float a, float b) const { return a - b; }
    __device__ void backward(float a, float b, float& ag, float& bg, float go) const {
        ag += go;
        bg -= go;
    }
};

struct Mul {
    static constexpr std::string_view name = "Mul";

    __device__ float forward(float a, float b) const { return a * b; }
    __device__ void backward(float a, float b, float& ag, float& bg, float go) const {
        ag += go * b;
        bg += go * a;
    }
};

struct Div {
    static constexpr std::string_view name = "Div";

    __device__ float forward(float a, float b) const { return a / b; }
    __device__ void backward(float a, float b, float& ag, float& bg, float go) const {
        ag += go / b;
        bg += -go * a / (b * b);
    }
};

} // namespace sorei::nn::binary
