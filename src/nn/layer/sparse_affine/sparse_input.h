#pragma once

#include "../layer.h"

namespace nn::layer {

class SparseInput : public TypedLayer<int> {
  public:
    SparseInput(const data::Shape& shape, const std::string& name = "SparseInput")
        : TypedLayer<int>(name),
          shape_(shape) {}

    void resize(const data::Shape& shape) { shape_ = shape; }
    data::Shape shape() const override { return shape_; }

  private:
    data::Shape shape_;
};

} // namespace nn::layer
