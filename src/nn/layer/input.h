#pragma once

#include "layer.h"

namespace nn::layer {

class Input : public TypedLayer<float> {
  public:
    Input(const data::Shape& shape, const std::string& name = "Input")
        : TypedLayer<float>(name),
          shape_(shape) {}

    void resize(const data::Shape& shape) { shape_ = shape; }
    data::Shape shape() const override { return shape_; }

  private:
    data::Shape shape_;
};

} // namespace nn::layer
