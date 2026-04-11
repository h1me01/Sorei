#pragma once

#include "layer.h"

namespace nn::layer {

class Affine : public TypedLayer<float> {
  public:
    Affine(Layer* input, Layer* weight, Layer* bias)
        : TypedLayer<float>("Affine"),
          input_(layer_cast<TypedLayer<float>>(input)),
          weight_(layer_cast<TypedLayer<float>>(weight)),
          bias_(layer_cast<TypedLayer<float>>(bias)) {

        CHECK(weight_->shape().cols() == input_->shape().rows());
    }

    void forward() override {
        kernel::mat_mul_forward(weight_->data(), input_->data(), data());
        kernel::elemwise_binary_broadcast_forward(
            bias_->data(), data(), data(), kernel::AddBinary{}
        );
    }

    void backward() override {
        tensor::GPUMatrix<float> tmp;
        kernel::elemwise_binary_broadcast_backward(
            bias_->data(), bias_->grad(), data(), tmp, grad(), kernel::AddBinary{}
        );
        kernel::mat_mul_backward(
            weight_->data(), weight_->grad(), input_->data(), input_->grad(), grad()
        );
    }

    std::vector<LayerInputSlot> mutable_inputs() override {
        return {
            LayerInputSlot::from(input_), LayerInputSlot::from(weight_), LayerInputSlot::from(bias_)
        };
    }

    tensor::Shape shape() const override { return {weight_->shape().rows(), input_->shape().cols()}; }

  private:
    TypedLayer<float>* input_;
    TypedLayer<float>* weight_;
    TypedLayer<float>* bias_;
};

} // namespace nn::layer
