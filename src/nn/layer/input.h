#pragma once

#include "layer.h"

namespace nn::layer {

template <typename T>
class Input : public TypedLayer<T> {
  public:
    Input(const data::Shape& shape, const std::string& name = "Input")
        : TypedLayer<T>(name),
          shape_(shape) {}

    void resize(const data::Shape& shape) { shape_ = shape; }
    data::Shape shape() const override { return shape_; }

  private:
    data::Shape shape_;
};

using InputInt = Input<int>;
using InputFloat = Input<float>;

} // namespace nn::layer
