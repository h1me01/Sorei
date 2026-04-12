#pragma once

#include "../../layer.h"
#include "../common.h"

namespace sorei::nn::layer {

class ElemwiseUnary : public TypedLayer<float> {
  public:
    using Op = std::variant<
        cuda::Identity,
        cuda::AddScaleUnary,
        cuda::DivLeftUnary,
        cuda::Clamp,
        cuda::Abs,
        cuda::PowInt,
        cuda::PowFloat,
        cuda::ReLU,
        cuda::ClampedReLU,
        cuda::SquaredClampedReLU,
        cuda::Sigmoid>;

  public:
    ElemwiseUnary(Layer* input, Op op)
        : TypedLayer<float>(elemwise_op_name(op)),
          input_(layer_cast<TypedLayer<float>>(input)),
          op_(op) {}

    void forward() override { forward(input_->data(), data(), op_); }
    void backward() override { backward(input_->data(), input_->grad(), grad(), op_); }

    static void
    forward(const tensor::GPUMatrix<float>& in, tensor::GPUMatrix<float>& out, const Op& op);

    static void backward(
        tensor::GPUMatrix<float>& in,
        tensor::GPUMatrix<float>& in_g,
        const tensor::GPUMatrix<float>& out_g,
        const Op& op
    );

    Op op() const { return op_; }
    std::vector<LayerInputSlot> mutable_inputs() override { return {LayerInputSlot::from(input_)}; }
    tensor::Shape shape() const override { return input_->shape(); }

  private:
    TypedLayer<float>* input_;
    Op op_;
};

} // namespace sorei::nn::layer
