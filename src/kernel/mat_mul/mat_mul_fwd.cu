#include "mat_mul.h"

namespace sorei::kernel {

constexpr float alpha = 1.0f;
constexpr float beta = 0.0f;

void mat_mul_forward(
    const tensor::GPUMatrix<float>& weight,
    const tensor::GPUMatrix<float>& in,
    tensor::GPUMatrix<float>& out
) {
    SOREI_CHECK(in.cols() == out.cols());
    SOREI_CHECK(weight.rows() == out.rows());
    SOREI_CHECK(weight.cols() == in.rows());

    SOREI_CHECK(weight.data());
    SOREI_CHECK(in.data());
    SOREI_CHECK(out.data());

    cublas::sgemm(false, false, alpha, weight, in, beta, out);
}

} // namespace sorei::kernel
