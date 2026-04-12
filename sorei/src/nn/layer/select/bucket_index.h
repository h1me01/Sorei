#pragma once

#include "../layer.h"

namespace sorei::nn::layer {

class BucketIndex : public TypedLayer<int> {
  public:
    BucketIndex(int count, int size, const std::string& name = "BucketIndex")
        : TypedLayer<int>(name),
          count_(count),
          shape_(1, size) {

        SOREI_CHECK(count > 0);
    }

    void resize(int size) {
        SOREI_CHECK(size > 0);
        shape_ = tensor::Shape(1, size);
    }

    int count() const { return count_; }
    tensor::Shape shape() const override { return shape_; }

  private:
    int count_;
    tensor::Shape shape_;
};

} // namespace sorei::nn::layer
