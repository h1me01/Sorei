#pragma once

#include "../layer.h"

namespace sorei::nn::layer {

class Mean : public TypedLayer<float> {
  public:
    Mean(Layer* input)
        : TypedLayer<float>("Mean"),
          input_(layer_cast<TypedLayer<float>>(input)) {}

    void forward() override;
    void backward() override;

    std::vector<LayerInputSlot> mutable_inputs() override { return {LayerInputSlot::from(input_)}; }
    tensor::Shape shape() const override { return {1, 1}; }

  private:
    TypedLayer<float>* input_;
};

} // namespace sorei::nn::layer
