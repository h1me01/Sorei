#pragma once

#include "../../layer.h"
#include "../common.h"
#include "ops.h"

namespace sorei::nn {

class ElemwiseBinary : public TypedLayer<float> {
  public:
    using Op = std::variant<binary::Add, binary::Sub, binary::Mul, binary::Div>;

  public:
    ElemwiseBinary(Layer* input1, Layer* input2, Op op)
        : TypedLayer<float>(elemwise_op_name(op)),
          input1_(checked_cast<TypedLayer<float>>(input1)),
          input2_(checked_cast<TypedLayer<float>>(input2)),
          op_(op) {

        const auto s1 = input1_->shape();
        const auto s2 = input2_->shape();

        SOREI_CHECK(s1.rows() == s2.rows());
        SOREI_CHECK(s1.cols() == s2.cols() || s1.cols() == 0 || s2.cols() == 0);
    }

    void forward() override;
    void backward() override;

    std::vector<LayerInputSlot> mutable_inputs() override {
        return {LayerInputSlot::from(input1_), LayerInputSlot::from(input2_)};
    }

    matrix::Shape shape() const override {
        return input1_->shape().cols() == 1 ? input2_->shape() : input1_->shape();
    }

  private:
    TypedLayer<float>* input1_;
    TypedLayer<float>* input2_;
    Op op_;
};

} // namespace sorei::nn
