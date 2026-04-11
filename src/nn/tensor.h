#pragma once

#include <vector>

#include "../tensor/include.h"

namespace nn {

// column-major 1D or 2D tensor
template <typename T>
class Tensor {
  public:
    Tensor() = default;

    Tensor(std::vector<int> shape)
        : shape_(shape),
          data_(size()) {
        if (shape.size() > 2)
            error("Tensor: Only 1D or 2D tensors are supported");
    }

    T operator[](int i) const {
        CHECK(shape_.size() == 1);
        CHECK(i >= 0 && i < shape_[0]);
        return data_[i];
    }

    T& operator[](int i) {
        CHECK(shape_.size() == 1);
        CHECK(i >= 0 && i < shape_[0]);
        return data_[i];
    }

    T operator()(int row, int col) const {
        CHECK(shape_.size() == 2);
        CHECK(row >= 0 && row < shape_[0]);
        CHECK(col >= 0 && col < shape_[1]);
        return data_[row + col * shape_[0]];
    }

    T& operator()(int row, int col) {
        CHECK(shape_.size() == 2);
        CHECK(row >= 0 && row < shape_[0]);
        CHECK(col >= 0 && col < shape_[1]);
        return data_[row + col * shape_[0]];
    }

    int size() const {
        if (shape_.empty())
            return 0;
        int s = 1;
        for (int dim : shape_)
            s *= dim;
        return s;
    }

    void fill(T value) { data_.fill(value); }

    void resize(std::vector<int> shape) {
        if (shape.size() > 2)
            error("Tensor: Only 1D or 2D tensors are supported");
        shape_ = shape;
        data_.resize(size());
    }

    tensor::PinnedCPUArray<T>& data() { return data_; }
    const tensor::PinnedCPUArray<T>& data() const { return data_; }

    std::vector<int> shape() const { return shape_; }

  private:
    std::vector<int> shape_;
    tensor::PinnedCPUArray<T> data_;
};

} // namespace nn