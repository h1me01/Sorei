#pragma once

#include <cuda_runtime.h>

#include "base.h"

namespace sorei::matrix {

template <typename T>
class HostPinnedMatrix : public HostMatrixBase<T, HostPinnedMatrix<T>> {
  public:
    HostPinnedMatrix()
        : shape_(0, 0),
          data_(nullptr) {}

    explicit HostPinnedMatrix(const Shape& shape)
        : shape_(shape),
          data_(nullptr) {
        alloc();
    }

    HostPinnedMatrix(int rows, int cols)
        : HostPinnedMatrix(Shape{rows, cols}) {}

    HostPinnedMatrix(const HostPinnedMatrix& o)
        : shape_(o.shape_),
          data_(nullptr) {
        alloc();
        std::copy(o.data_, o.data_ + o.size(), data_);
    }

    HostPinnedMatrix(HostPinnedMatrix&& o) noexcept
        : shape_(o.shape_),
          data_(o.data_) {
        o.shape_ = Shape{0, 0};
        o.data_ = nullptr;
    }

    HostPinnedMatrix& operator=(const HostPinnedMatrix& o) {
        if (this != &o) {
            if (shape_ != o.shape_) {
                free();
                shape_ = o.shape_;
                alloc();
            }
            std::copy(o.data_, o.data_ + o.size(), data_);
        }
        return *this;
    }

    HostPinnedMatrix& operator=(HostPinnedMatrix&& o) noexcept {
        if (this != &o) {
            free();
            shape_ = o.shape_;
            data_ = o.data_;
            o.shape_ = Shape{0, 0};
            o.data_ = nullptr;
        }
        return *this;
    }

    ~HostPinnedMatrix() { free(); }

    static HostPinnedMatrix zeros(const Shape& shape) {
        HostPinnedMatrix m(shape);
        m.clear();
        return m;
    }

    static HostPinnedMatrix filled(const Shape& shape, const T& value) {
        HostPinnedMatrix m(shape);
        m.fill(value);
        return m;
    }

    int rows() const { return shape_.rows(); }
    int cols() const { return shape_.cols(); }
    int size() const { return shape_.size(); }
    Shape shape() const { return shape_; }
    size_t bytes() const { return static_cast<size_t>(size()) * sizeof(T); }

    T* data() const { return data_; }

    void resize(const Shape& new_shape) {
        if (new_shape == shape_)
            return;
        free();
        shape_ = new_shape;
        alloc();
    }

  private:
    Shape shape_;
    T* data_;

    void alloc() {
        if (size() > 0)
            SOREI_CUDA_CHECK(cudaMallocHost(&data_, bytes()));
    }

    void free() {
        if (data_) {
            SOREI_CUDA_CHECK(cudaFreeHost(data_));
            data_ = nullptr;
        }
    }
};

} // namespace sorei::matrix