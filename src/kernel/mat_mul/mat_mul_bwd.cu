#include "mat_mul.h"

namespace kernel {

constexpr float alpha = 1.0f;
constexpr float beta = 1.0f;

void mat_mul_backward(
    const data::GPUMatrix<float>& weight,
    data::GPUMatrix<float>& weight_g,
    const data::GPUMatrix<float>& in,
    data::GPUMatrix<float>& in_g,
    const data::GPUMatrix<float>& out_g
) {
    CHECK(weight.rows() == out_g.rows());

    CHECK(weight.data());
    CHECK(in.data());

    if (!weight_g.empty())
        cublas::sgemm(false, true, alpha, out_g, in, beta, weight_g);

    if (!in_g.empty()) {
        CHECK(in_g.cols() == out_g.cols());
        CHECK(in_g.rows() == weight.cols());
        cublas::sgemm(true, false, alpha, weight, out_g, beta, in_g);
    }
}

} // namespace kernel
