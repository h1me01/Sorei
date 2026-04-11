#pragma once

#include "../layer.h"

namespace nn::layer {

class BucketIndex : public TypedLayer<int> {
  public:
    BucketIndex(int count, int size, const std::string& name = "BucketIndex")
        : TypedLayer<int>(name),
          count_(count),
          shape_(1, size) {

        CHECK(count > 0);
    }

    void resize(int size) {
        CHECK(size > 0);
        shape_ = data::Shape(1, size);
    }

    int count() const { return count_; }
    data::Shape shape() const override { return shape_; }

  private:
    int count_;
    data::Shape shape_;
};

} // namespace nn::layer
