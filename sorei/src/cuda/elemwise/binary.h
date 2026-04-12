#pragma once

#include <string_view>

namespace sorei::cuda {

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

} // namespace sorei::cuda
