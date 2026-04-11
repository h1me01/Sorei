#pragma once

#include "../layer.h"

namespace nn::layer {

class PairwiseMul : public TypedLayer<float> {
  public:
    PairwiseMul(Layer* input)
        : TypedLayer<float>("PairwiseMul"),
          input_(layer_cast<TypedLayer<float>>(input)) {

        CHECK(input_->shape().rows() % 2 == 0);
    }

    void forward() override;
    void backward() override;

    std::vector<LayerInputSlot> mutable_inputs() override { return {LayerInputSlot::from(input_)}; }

    tensor::Shape shape() const override {
        auto input_shape = input_->shape();
        return {input_shape.rows() / 2, input_shape.cols()};
    }

  private:
    TypedLayer<float>* input_;
};

} // namespace nn::layer