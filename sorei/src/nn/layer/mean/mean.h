#pragma once

#include "../layer.h"

namespace sorei::nn {

class Mean : public TypedLayer<float> {
  public:
    Mean(Layer* input)
        : TypedLayer<float>("Mean"),
          input_(checked_cast<TypedLayer<float>>(input)) {}

    void forward() override;
    void backward() override;

    std::vector<LayerInputSlot> mutable_inputs() override { return {LayerInputSlot::from(input_)}; }
    matrix::Shape shape() const override { return {1, 1}; }

  private:
    TypedLayer<float>* input_;
};

} // namespace sorei::nn
