#pragma once

#include "../../layer.h"
#include "../common.h"

namespace sorei::nn::layer {

class ElemwiseBinary : public TypedLayer<float> {
  public:
    using Op = std::variant<cuda::AddBinary, cuda::SubBinary, cuda::MulBinary, cuda::DivBinary>;

  public:
    ElemwiseBinary(Layer* input1, Layer* input2, Op op)
        : TypedLayer<float>(elemwise_op_name(op)),
          input1_(layer_cast<TypedLayer<float>>(input1)),
          input2_(layer_cast<TypedLayer<float>>(input2)),
          op_(op) {

        const auto s1 = input1_->shape();
        const auto s2 = input2_->shape();

        if (broadcast()) {
            SOREI_CHECK(s1.rows() == s2.rows());
            SOREI_CHECK(s1.cols() == 1 || s2.cols() == 1);
        } else {
            SOREI_CHECK(s1.rows() == s2.rows());
            SOREI_CHECK(s1.cols() == s2.cols() || s1.cols() == 0 || s2.cols() == 0);
        }
    }

    void forward() override {
        if (broadcast())
            broadcast_forward(input1_->data(), input2_->data(), data(), op_);
        else
            forward(input1_->data(), input2_->data(), data(), op_);
    }

    void backward() override {
        if (broadcast()) {
            broadcast_backward(
                input1_->data(), input1_->grad(), input2_->data(), input2_->grad(), grad(), op_
            );
        } else {
            backward(
                input1_->data(), input1_->grad(), input2_->data(), input2_->grad(), grad(), op_
            );
        }
    }

    static void forward(
        const tensor::DeviceMatrix<float>& a,
        const tensor::DeviceMatrix<float>& b,
        tensor::DeviceMatrix<float>& c,
        const Op& op
    );

    static void backward(
        const tensor::DeviceMatrix<float>& a,
        tensor::DeviceMatrix<float>& a_g,
        const tensor::DeviceMatrix<float>& b,
        tensor::DeviceMatrix<float>& b_g,
        const tensor::DeviceMatrix<float>& c_g,
        const Op& op
    );

    static void broadcast_forward(
        const tensor::DeviceMatrix<float>& a,
        const tensor::DeviceMatrix<float>& b,
        tensor::DeviceMatrix<float>& c,
        const Op& op
    );

    static void broadcast_backward(
        const tensor::DeviceMatrix<float>& a,
        tensor::DeviceMatrix<float>& a_g,
        const tensor::DeviceMatrix<float>& b,
        tensor::DeviceMatrix<float>& b_g,
        const tensor::DeviceMatrix<float>& c_g,
        const Op& op
    );

    std::vector<LayerInputSlot> mutable_inputs() override {
        return {LayerInputSlot::from(input1_), LayerInputSlot::from(input2_)};
    }

    tensor::Shape shape() const override {
        return input1_->shape().cols() == 1 ? input2_->shape() : input1_->shape();
    }

  private:
    TypedLayer<float>* input1_;
    TypedLayer<float>* input2_;
    Op op_;

    bool broadcast() const { return input1_->shape() != input2_->shape(); }
};

} // namespace sorei::nn::layer
