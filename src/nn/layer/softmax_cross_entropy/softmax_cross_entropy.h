#pragma once

#include <bit>

#include "../input.h"
#include "../layer.h"

namespace nn::layer {

class SoftmaxCrossEntropy : public TypedLayer<float> {
  public:
    SoftmaxCrossEntropy(Layer* logits, InputInt* labels)
        : TypedLayer<float>("SoftmaxCrossEntropy"),
          input_(layer_cast<TypedLayer<float>>(logits)),
          labels_(labels) {

        CHECK(labels_);
    }

    void forward() override;
    void backward() override;

    std::vector<LayerInputSlot> mutable_inputs() override {
        return {LayerInputSlot::from(input_), LayerInputSlot::from(labels_)};
    }

    tensor::Shape shape() const override { return {1, input_->shape().cols()}; }

  private:
    TypedLayer<float>* input_;
    InputInt* labels_;
    tensor::GPUMatrix<float> probs_;

    int get_block_size() const {
        unsigned int n = input_->shape().rows();
        if (n <= 32)
            return 32;
        return (int)std::min(std::bit_ceil(n), 1024u);
    }
};

} // namespace nn::layer
