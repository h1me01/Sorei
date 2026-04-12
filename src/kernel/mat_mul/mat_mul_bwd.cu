#include "mat_mul.h"

namespace sorei::kernel {

constexpr float alpha = 1.0f;
constexpr float beta = 1.0f;

void mat_mul_backward(
    const tensor::GPUMatrix<float>& weight,
    tensor::GPUMatrix<float>& weight_g,
    const tensor::GPUMatrix<float>& in,
    tensor::GPUMatrix<float>& in_g,
    const tensor::GPUMatrix<float>& out_g
) {
    SOREI_CHECK(weight.rows() == out_g.rows());

    SOREI_CHECK(weight.data());
    SOREI_CHECK(in.data());

    if (!weight_g.empty())
        cublas::sgemm(false, true, alpha, out_g, in, beta, weight_g);

    if (!in_g.empty()) {
        SOREI_CHECK(in_g.cols() == out_g.cols());
        SOREI_CHECK(in_g.rows() == weight.cols());
        cublas::sgemm(true, false, alpha, weight, out_g, beta, in_g);
    }
}

} // namespace sorei::kernel
