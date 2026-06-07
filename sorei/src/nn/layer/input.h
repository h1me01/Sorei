#pragma once

#include "layer.h"

namespace sorei::nn {

template <typename T>
class Input : public TypedLayer<T> {
  public:
    Input(const std::string& name, const matrix::Shape& shape)
        : TypedLayer<T>(name),
          shape_(shape) {}

    void resize(const matrix::Shape& shape) { shape_ = shape; }
    matrix::Shape shape() const override { return shape_; }

  private:
    matrix::Shape shape_;
};

using InputInt = Input<int>;
using InputFloat = Input<float>;

} // namespace sorei::nn
