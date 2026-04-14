#pragma once

#include <variant>

#include "../concat/concat.h"
#include "../input.h"
#include "../layer.h"

namespace sorei::nn::layer {

using ActOp = std::
    variant<cuda::Identity, cuda::ReLU, cuda::ClampedReLU, cuda::SquaredClampedReLU, cuda::Sigmoid>;

class SparseAffineBase : public TypedLayer<float> {
  public:
    SparseAffineBase(InputInt* input, Layer* weight, Layer* bias)
        : TypedLayer<float>("SparseAffine"),
          input_(input),
          weight_(layer_cast<TypedLayer<float>>(weight)),
          bias_(layer_cast<TypedLayer<float>>(bias)) {

        SOREI_CHECK(input);
    }

    void fuse_with_concat(FusedConcat* c) {
        SOREI_CHECK(c);
        concat_ = c;
        out_offset_ = c->offset_of(this);
        // concat holds out buffer
        drop_buffers();
    }

    void set_activation(ActOp act_op) { act_op_ = act_op; }
    ActOp activation() const { return act_op_; }
    bool has_activation() const { return !std::holds_alternative<cuda::Identity>(act_op_); }

    InputInt* input() const { return input_; }
    Layer* weight() const { return weight_; }
    Layer* bias() const { return bias_; }

    std::vector<LayerInputSlot> mutable_inputs() override {
        return {
            LayerInputSlot::from(input_), LayerInputSlot::from(weight_), LayerInputSlot::from(bias_)
        };
    }

    tensor::Shape shape() const override {
        return {weight_->shape().rows(), input_->shape().cols()};
    }

  protected:
    int out_offset_ = 0;
    FusedConcat* concat_ = nullptr;
    ActOp act_op_ = cuda::Identity{};

    InputInt* input_;
    TypedLayer<float>* weight_;
    TypedLayer<float>* bias_;

    tensor::DeviceMatrix<float>& effective_data() {
        return concat_ ? concat_->data() : TypedLayer<float>::data();
    }

    tensor::DeviceMatrix<float>& effective_grad() {
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

    tensor::Shape shape() const override {
        return {weight_->shape().rows() / 2, input_->shape().cols()};
    }
};

} // namespace sorei::nn::layer
