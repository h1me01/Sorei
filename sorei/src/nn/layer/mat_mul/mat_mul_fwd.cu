#include "mat_mul.h"

namespace sorei::nn::layer {

constexpr float alpha = 1.0f;
constexpr float beta = 0.0f;

void MatMul::forward(
    const tensor::DeviceMatrix<float>& weight,
    const tensor::DeviceMatrix<float>& in,
    tensor::DeviceMatrix<float>& out
) {
    SOREI_CHECK(in.cols() == out.cols());
    SOREI_CHECK(weight.rows() == out.rows());
    SOREI_CHECK(weight.cols() == in.rows());

    SOREI_CHECK(weight.data());
    SOREI_CHECK(in.data());
    SOREI_CHECK(out.data());

    cuda::cublas::sgemm(false, false, alpha, weight, in, beta, out);
}

} // namespace sorei::nn::layer
