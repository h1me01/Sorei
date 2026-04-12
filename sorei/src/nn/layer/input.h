#pragma once

#include "layer.h"

namespace sorei::nn::layer {

template <typename T>
class Input : public TypedLayer<T> {
  public:
    Input(const tensor::Shape& shape, const std::string& name = "Input")
        : TypedLayer<T>(name),
          shape_(shape) {}

    void resize(const tensor::Shape& shape) { shape_ = shape; }
    tensor::Shape shape() const override { return shape_; }

  private:
    tensor::Shape shape_;
};

using InputInt = Input<int>;
using InputFloat = Input<float>;

} // namespace sorei::nn::layer
