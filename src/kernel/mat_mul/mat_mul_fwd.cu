#include "mat_mul.h"

namespace kernel {

constexpr float alpha = 1.0f;
constexpr float beta = 0.0f;

void mat_mul_forward(
    const tensor::GPUMatrix<float>& weight,
    const tensor::GPUMatrix<float>& in,
    tensor::GPUMatrix<float>& out
) {
    CHECK(in.cols() == out.cols());
    CHECK(weight.rows() == out.rows());
    CHECK(weight.cols() == in.rows());

    CHECK(weight.data());
    CHECK(in.data());
    CHECK(out.data());

    cublas::sgemm(false, false, alpha, weight, in, beta, out);
}

} // namespace kernel
