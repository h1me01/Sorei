#pragma once

#include "../../layer.h"
#include "../common.h"
#include "ops.h"

namespace sorei::nn {

class ElemwiseUnary : public TypedLayer<float> {
  public:
    using Op = std::variant<
        unary::Identity,
        unary::AddScale,
        unary::DivLeft,
        unary::Clamp,
        unary::Abs,
        unary::ReLU,
        unary::ClampedReLU,
        unary::SquaredClampedReLU,
        unary::Sigmoid>;

  public:
    ElemwiseUnary(Layer* input, Op op)
        : TypedLayer<float>(elemwise_op_name(op)),
          input_(checked_cast<TypedLayer<float>>(input)),
          op_(op) {}

    void forward() override;
    void backward() override;

    Op op() const { return op_; }
    std::vector<LayerInputSlot> mutable_inputs() override { return {LayerInputSlot::from(input_)}; }
    matrix::Shape shape() const override { return input_->shape(); }

  private:
    TypedLayer<float>* input_;
    Op op_;
};

} // namespace sorei::nn
