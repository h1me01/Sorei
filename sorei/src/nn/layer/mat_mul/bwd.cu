#include "mat_mul.h"

namespace sorei::nn {

constexpr float alpha = 1.0f;

void MatMul::backward() {
    auto& weight = weight_->data();
    auto& weight_g = weight_->grad();
    auto& in = input_->data();
    auto& in_g = input_->grad();
    auto& out_g = grad();

    SOREI_CHECK(weight.rows() == out_g.rows());

    SOREI_CHECK(weight.data());
    SOREI_CHECK(in.data());
    SOREI_CHECK(out_g.data());

    if (!weight_g.empty()) {
        cublas::sgemm(
            false, true, alpha, out_g, in, weight_->consume_grad_write() ? 0.0f : 1.0f, weight_g
        );
    }

    if (!in_g.empty()) {
        SOREI_CHECK(in_g.cols() == out_g.cols());
        SOREI_CHECK(in_g.rows() == weight.cols());
        cublas::sgemm(
            true, false, alpha, weight, out_g, input_->consume_grad_write() ? 0.0f : 1.0f, in_g
        );
    }
}

} // namespace sorei::nn
