#pragma once

#include "elemwise/binary/binary.h"
#include "layer.h"
#include "mat_mul/mat_mul.h"

namespace sorei::nn {

class Affine : public TypedLayer<float> {
  public:
    Affine(Layer* input, Layer* weight, Layer* bias)
        : TypedLayer<float>("Affine"),
          input_(checked_cast<TypedLayer<float>>(input)),
          weight_(checked_cast<TypedLayer<float>>(weight)),
          bias_(checked_cast<TypedLayer<float>>(bias)) {

        SOREI_CHECK(weight_->shape().cols() == input_->shape().rows());
    }

    void forward() override {
        MatMul::forward(weight_->data(), input_->data(), data());
        ElemwiseBinary::broadcast_forward(
            bias_->data(), data(), data(), ElemwiseBinary::Op{binary::Add{}}
        );
    }

    void backward() override {
        const bool ow_bias = bias_->consume_grad_write();
        const bool ow_input = input_->consume_grad_write();
        const bool ow_weight = weight_->consume_grad_write();

        matrix::DeviceMatrix<float> tmp;
        ElemwiseBinary::broadcast_backward(
            bias_->data(),
            bias_->grad(),
            data(),
            tmp,
            grad(),
            ElemwiseBinary::Op{binary::Add{}},
            ow_bias
        );
        MatMul::backward(
            weight_->data(),
            weight_->grad(),
            input_->data(),
            input_->grad(),
            grad(),
            ow_input,
            ow_weight
        );
    }

    std::vector<LayerInputSlot> mutable_inputs() override {
        return {
            LayerInputSlot::from(input_), LayerInputSlot::from(weight_), LayerInputSlot::from(bias_)
        };
    }

    matrix::Shape shape() const override {
        return {weight_->shape().rows(), input_->shape().cols()};
    }

  private:
    TypedLayer<float>* input_;
    TypedLayer<float>* weight_;
    TypedLayer<float>* bias_;
};

} // namespace sorei::nn
