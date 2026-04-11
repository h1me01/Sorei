#pragma once

#include "layer.h"

namespace nn::layer {

class ElemwiseUnary : public TypedLayer<float> {
  public:
    ElemwiseUnary(Layer* input, kernel::UnaryOp op)
        : TypedLayer<float>(kernel::elemwise_op_name(op)),
          input_(layer_cast<TypedLayer<float>>(input)),
          op_(op) {}

    void forward() override { kernel::elemwise_unary_forward(input_->data(), data(), op_); }

    void backward() override {
        kernel::elemwise_unary_backward(input_->data(), input_->grad(), grad(), op_);
    }

    kernel::UnaryOp op() const { return op_; }
    std::vector<LayerInputSlot> mutable_inputs() override { return {LayerInputSlot::from(input_)}; }
    data::Shape shape() const override { return input_->shape(); }

  private:
    TypedLayer<float>* input_;
    kernel::UnaryOp op_;
};

} // namespace nn::layer
