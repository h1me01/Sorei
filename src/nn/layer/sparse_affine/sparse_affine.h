#pragma once

#include <variant>

#include "../concat/concat.h"
#include "../input.h"
#include "../layer.h"

namespace nn::layer {

class SparseAffineBase : public TypedLayer<float> {
  public:
    SparseAffineBase(InputInt* input, Layer* weight, Layer* bias)
        : TypedLayer<float>("SparseAffine"),
          input_(input),
          weight_(layer_cast<TypedLayer<float>>(weight)),
          bias_(layer_cast<TypedLayer<float>>(bias)) {

        CHECK(input);
    }

    void fuse_with_concat(FusedConcat* c) {
        CHECK(c);
        concat_ = c;
        out_offset_ = c->offset_of(this);
        // concat holds out buffer
        drop_buffers();
    }

    void set_activation(kernel::ActOp act_op) { act_op_ = act_op; }
    kernel::ActOp activation() const { return act_op_; }
    bool has_activation() const { return !std::holds_alternative<kernel::Identity>(act_op_); }

    InputInt* input() const { return input_; }
    Layer* weight() const { return weight_; }
    Layer* bias() const { return bias_; }

    std::vector<LayerInputSlot> mutable_inputs() override {
        return {
            LayerInputSlot::from(input_), LayerInputSlot::from(weight_), LayerInputSlot::from(bias_)
        };
    }

    data::Shape shape() const override { return {weight_->shape().rows(), input_->shape().cols()}; }

  protected:
    int out_offset_ = 0;
    FusedConcat* concat_ = nullptr;
    kernel::ActOp act_op_ = kernel::Identity{};

    InputInt* input_;
    TypedLayer<float>* weight_;
    TypedLayer<float>* bias_;

    data::GPUMatrix<float>& effective_data() {
        return concat_ ? concat_->data() : TypedLayer<float>::data();
    }

    data::GPUMatrix<float>& effective_grad() {
        return concat_ ? concat_->grad() : TypedLayer<float>::grad();
    }
};

struct SparseAffine : SparseAffineBase {
    SparseAffine(InputInt* input, Layer* weight, Layer* bias)
        : SparseAffineBase(input, weight, bias) {}

    void forward() override;
    void backward() override;
};

struct SparseAffinePairwiseMul : SparseAffineBase {
    SparseAffinePairwiseMul(InputInt* input, Layer* weight, Layer* bias)
        : SparseAffineBase(input, weight, bias) {}

    void forward() override;
    void backward() override;

    data::Shape shape() const override {
        return {weight_->shape().rows() / 2, input_->shape().cols()};
    }
};

} // namespace nn::layer
