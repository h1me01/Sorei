#pragma once

#include "../layer.h"

namespace sorei::nn::layer {

class MatMul : public TypedLayer<float> {
  public:
    MatMul(Layer* weight, Layer* input)
        : TypedLayer<float>("MatMul"),
          weight_(layer_cast<TypedLayer<float>>(weight)),
          input_(layer_cast<TypedLayer<float>>(input)) {

        SOREI_CHECK(weight_->shape().cols() == input_->shape().rows());
    }

    void forward() override { forward(weight_->data(), input_->data(), data()); }

    void backward() override {
        backward(weight_->data(), weight_->grad(), input_->data(), input_->grad(), grad());
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
        const matrix::DeviceMatrix<float>& out_g
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

} // namespace sorei::nn::layer
