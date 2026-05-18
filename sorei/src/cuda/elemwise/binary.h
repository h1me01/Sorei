#pragma once

#include <string_view>

namespace sorei::cuda {

struct AddBinary {
    static constexpr std::string_view name = "Add";

    __device__ float forward(float a, float b) const { return a + b; }
    __device__ void backward(float a, float b, float& ag, float& bg, float go) const {
        ag += go;
        bg += go;
    }
};

struct SubBinary {
    static constexpr std::string_view name = "Sub";

    __device__ float forward(float a, float b) const { return a - b; }
    __device__ void backward(float a, float b, float& ag, float& bg, float go) const {
        ag += go;
        bg -= go;
    }
};

struct MulBinary {
    static constexpr std::string_view name = "Mul";

    __device__ float forward(float a, float b) const { return a * b; }
    __device__ void backward(float a, float b, float& ag, float& bg, float go) const {
        ag += go * b;
        bg += go * a;
    }
};

struct DivBinary {
    static constexpr std::string_view name = "Div";

    __device__ float forward(float a, float b) const { return a / b; }
    __device__ void backward(float a, float b, float& ag, float& bg, float go) const {
        ag += go / b;
        bg += -go * a / (b * b);
    }
};

} // namespace sorei::cuda
