#pragma once

#include "layer.h"

namespace nn::layer {

class ElemwiseBinary : public TypedLayer<float> {
  public:
    ElemwiseBinary(Layer* input1, Layer* input2, kernel::BinaryOp op)
        : TypedLayer<float>(kernel::elemwise_op_name(op)),
          input1_(layer_cast<TypedLayer<float>>(input1)),
          input2_(layer_cast<TypedLayer<float>>(input2)),
          op_(op) {

        const auto s1 = input1_->shape();
        const auto s2 = input2_->shape();

        if (broadcast()) {
            CHECK(s1.rows() == s2.rows());
            CHECK(s1.cols() == 1 || s2.cols() == 1);
        } else {
            CHECK(s1.rows() == s2.rows());
            CHECK(s1.cols() == s2.cols() || s1.cols() == 0 || s2.cols() == 0);
        }
    }

    void forward() override {
        if (broadcast()) {
            kernel::elemwise_binary_broadcast_forward(
                input1_->data(), input2_->data(), data(), op_
            );
        } else {
            kernel::elemwise_binary_forward(input1_->data(), input2_->data(), data(), op_);
        }
    }

    void backward() override {
        if (broadcast()) {
            kernel::elemwise_binary_broadcast_backward(
                input1_->data(), input1_->grad(), input2_->data(), input2_->grad(), grad(), op_
            );
        } else {
            kernel::elemwise_binary_backward(
                input1_->data(), input1_->grad(), input2_->data(), input2_->grad(), grad(), op_
            );
        }
    }

    std::vector<LayerInputSlot> mutable_inputs() override {
        return {LayerInputSlot::from(input1_), LayerInputSlot::from(input2_)};
    }

    tensor::Shape shape() const override {
        return input1_->shape().cols() == 1 ? input2_->shape() : input1_->shape();
    }

  private:
    TypedLayer<float>* input1_;
    TypedLayer<float>* input2_;
    kernel::BinaryOp op_;

    bool broadcast() const { return input1_->shape() != input2_->shape(); }
};

} // namespace nn::layer
