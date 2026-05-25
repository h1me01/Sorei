#include "mat_mul.h"

namespace sorei::nn {

constexpr float alpha = 1.0f;

void MatMul::backward(
    const matrix::DeviceMatrix<float>& weight,
    matrix::DeviceMatrix<float>& weight_g,
    const matrix::DeviceMatrix<float>& in,
    matrix::DeviceMatrix<float>& in_g,
    const matrix::DeviceMatrix<float>& out_g,
    bool ow_in_g,
    bool ow_weight_g
) {
    SOREI_CHECK(weight.rows() == out_g.rows());

    SOREI_CHECK(weight.data());
    SOREI_CHECK(in.data());

    if (!weight_g.empty())
        cublas::sgemm(false, true, alpha, out_g, in, ow_weight_g ? 0.0f : 1.0f, weight_g);

    if (!in_g.empty()) {
        SOREI_CHECK(in_g.cols() == out_g.cols());
        SOREI_CHECK(in_g.rows() == weight.cols());
        cublas::sgemm(true, false, alpha, weight, out_g, ow_in_g ? 0.0f : 1.0f, in_g);
    }
}

} // namespace sorei::nn
