#pragma once

#include <memory>

#include "base.h"

namespace sorei::matrix {

template <typename T>
class HostMatrix : public HostMatrixBase<T, HostMatrix<T>> {
  public:
    HostMatrix()
        : shape_(0, 0) {}

    explicit HostMatrix(const Shape& shape)
        : shape_(shape),
          data_(shape.size() > 0 ? std::make_unique<T[]>(shape.size()) : nullptr) {}

    HostMatrix(int rows, int cols)
        : HostMatrix(Shape{rows, cols}) {}

    HostMatrix(const HostMatrix& o)
        : shape_(o.shape_),
          data_(o.size() > 0 ? std::make_unique<T[]>(o.size()) : nullptr) {
        std::copy(o.data_.get(), o.data_.get() + o.size(), data_.get());
    }

    HostMatrix(HostMatrix&& o) noexcept
        : shape_(o.shape_),
          data_(std::move(o.data_)) {
        o.shape_ = Shape{0, 0};
    }

    HostMatrix& operator=(const HostMatrix& o) {
        if (this != &o) {
            shape_ = o.shape_;
            data_ = o.size() > 0 ? std::make_unique<T[]>(o.size()) : nullptr;
            std::copy(o.data_.get(), o.data_.get() + o.size(), data_.get());
        }
        return *this;
    }

    HostMatrix& operator=(HostMatrix&& o) noexcept {
        if (this != &o) {
            shape_ = o.shape_;
            data_ = std::move(o.data_);
            o.shape_ = Shape{0, 0};
        }
        return *this;
    }

    ~HostMatrix() = default;

    static HostMatrix zeros(const Shape& shape) {
        HostMatrix m(shape);
        m.clear();
        return m;
    }

    static HostMatrix filled(const Shape& shape, const T& value) {
        HostMatrix m(shape);
        m.fill(value);
        return m;
    }

    int rows() const { return shape_.rows(); }
    int cols() const { return shape_.cols(); }
    int size() const { return shape_.size(); }
    Shape shape() const { return shape_; }
    size_t bytes() const { return static_cast<size_t>(size()) * sizeof(T); }

    T* data() const { return data_.get(); }

    void resize(const Shape& new_shape) {
        if (new_shape != shape_) {
            shape_ = new_shape;
            data_ = new_shape.size() > 0 ? std::make_unique<T[]>(new_shape.size()) : nullptr;
        }
    }

  private:
    Shape shape_;
    std::unique_ptr<T[]> data_;
};

} // namespace sorei::matrix