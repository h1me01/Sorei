#pragma once

#include "elemwise/binary/binary.h"
#include "layer.h"
#include "mat_mul/mat_mul.h"

namespace sorei::nn::layer {

class Affine : public TypedLayer<float> {
  public:
    Affine(Layer* input, Layer* weight, Layer* bias)
        : TypedLayer<float>("Affine"),
          input_(layer_cast<TypedLayer<float>>(input)),
          weight_(layer_cast<TypedLayer<float>>(weight)),
          bias_(layer_cast<TypedLayer<float>>(bias)) {

        SOREI_CHECK(weight_->shape().cols() == input_->shape().rows());
    }

    void forward() override {
        MatMul::forward(weight_->data(), input_->data(), data());
        ElemwiseBinary::broadcast_forward(
            bias_->data(), data(), data(), ElemwiseBinary::Op{cuda::AddBinary{}}
        );
    }

    void backward() override {
        tensor::DeviceMatrix<float> tmp;
        ElemwiseBinary::broadcast_backward(
            bias_->data(), bias_->grad(), data(), tmp, grad(), ElemwiseBinary::Op{cuda::AddBinary{}}
        );
        MatMul::backward(weight_->data(), weight_->grad(), input_->data(), input_->grad(), grad());
    }

    std::vector<LayerInputSlot> mutable_inputs() override {
        return {
            LayerInputSlot::from(input_), LayerInputSlot::from(weight_), LayerInputSlot::from(bias_)
        };
    }

    tensor::Shape shape() const override {
        return {weight_->shape().rows(), input_->shape().cols()};
    }

  private:
    TypedLayer<float>* input_;
    TypedLayer<float>* weight_;
    TypedLayer<float>* bias_;
};

} // namespace sorei::nn::layer
