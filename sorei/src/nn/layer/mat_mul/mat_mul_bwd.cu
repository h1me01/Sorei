#include "mat_mul.h"

namespace sorei::nn::layer {

constexpr float alpha = 1.0f;
constexpr float beta = 1.0f;

void MatMul::backward(
    const tensor::DeviceMatrix<float>& weight,
    tensor::DeviceMatrix<float>& weight_g,
    const tensor::DeviceMatrix<float>& in,
    tensor::DeviceMatrix<float>& in_g,
    const tensor::DeviceMatrix<float>& out_g
) {
    SOREI_CHECK(weight.rows() == out_g.rows());

    SOREI_CHECK(weight.data());
    SOREI_CHECK(in.data());

    if (!weight_g.empty())
        cuda::cublas::sgemm(false, true, alpha, out_g, in, beta, weight_g);

    if (!in_g.empty()) {
        SOREI_CHECK(in_g.cols() == out_g.cols());
        SOREI_CHECK(in_g.rows() == weight.cols());
        cuda::cublas::sgemm(true, false, alpha, weight, out_g, beta, in_g);
    }
}

} // namespace sorei::nn::layer
