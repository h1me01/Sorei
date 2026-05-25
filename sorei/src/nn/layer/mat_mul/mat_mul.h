#pragma once

#include "../../../cublas/cublas.h"
#include "../layer.h"

namespace sorei::nn {

class MatMul : public TypedLayer<float> {
  public:
    MatMul(Layer* weight, Layer* input)
        : TypedLayer<float>("MatMul"),
          weight_(checked_cast<TypedLayer<float>>(weight)),
          input_(checked_cast<TypedLayer<float>>(input)) {

        SOREI_CHECK(weight_->shape().cols() == input_->shape().rows());
    }

    void forward() override { forward(weight_->data(), input_->data(), data()); }

    void backward() override {
        const bool ow_input = input_->consume_grad_write();
        const bool ow_weight = weight_->consume_grad_write();

        backward(
            weight_->data(),
            weight_->grad(),
            input_->data(),
            input_->grad(),
            grad(),
            ow_input,
            ow_weight
        );
    }

    static void forward(
        const matrix::DeviceMatrix<float>& weight,
        const matrix::DeviceMatrix<float>& in,
        matrix::DeviceMatrix<float>& out
    );

    static void backward(
        const matrix::DeviceMatrix<float>& weight,
        matrix::DeviceMatrix<float>& weight_g,
        const matrix::DeviceMatrix<float>& in,
        matrix::DeviceMatrix<float>& in_g,
        const matrix::DeviceMatrix<float>& out_g,
        bool ow_in_g = false,
        bool ow_weight_g = false
    );

    std::vector<LayerInputSlot> mutable_inputs() override {
        return {LayerInputSlot::from(weight_), LayerInputSlot::from(input_)};
    }

    matrix::Shape shape() const override {
        return {weight_->shape().rows(), input_->shape().cols()};
    }

  private:
    TypedLayer<float>* weight_;
    TypedLayer<float>* input_;
};

} // namespace sorei::nn
