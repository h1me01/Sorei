#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>

#include "../misc.h"
#include "array.h"
#include "shape.h"

namespace sorei::tensor {

template <typename T, typename Storage>
class MatrixBase {
  public:
    MatrixBase()
        : shape_(0, 0) {}

    explicit MatrixBase(const Shape& shape)
        : shape_(shape),
          data_(shape.rows() * shape.cols()) {}

    int rows() const { return shape_.rows(); }
    int cols() const { return shape_.cols(); }
    int size() const { return data_.size(); }
    size_t bytes() const { return data_.bytes(); }
    Shape shape() const { return shape_; }
    bool empty() const { return size() == 0; }

    T* data() const { return data_.data(); }
    void clear() { data_.clear(); }

    T operator()(int i) const { return data_[i]; }
    T& operator()(int i) { return data_[i]; }

    T operator()(int r, int c) const {
        SOREI_CHECK(in_bounds(r, c));
        return data_[rows() * c + r];
    }

    T& operator()(int r, int c) {
        SOREI_CHECK(in_bounds(r, c));
        return data_[rows() * c + r];
    }

    T* begin() { return data_.data(); }
    T* end() { return data_.data() + size(); }
    const T* begin() const { return data_.data(); }
    const T* end() const { return data_.data() + size(); }

  protected:
    Shape shape_;
    Storage data_;

    bool in_bounds(int r, int c) const { return r >= 0 && r < rows() && c >= 0 && c < cols(); }
};

template <typename T, typename Storage = CPUArray<T>>
class CPUMatrix : public MatrixBase<T, Storage> {
    using Base = MatrixBase<T, Storage>;

  public:
    CPUMatrix() = default;

    explicit CPUMatrix(const Shape& shape)
        : Base(shape) {}

    void fill(T value) { this->data_.fill(value); }

    // no-op if shape = shape()
    void resize(const Shape& shape) {
        if (shape == this->shape_)
            return;
        this->shape_ = shape;
        this->data_ = Storage(shape.size());
        this->clear();
    }

    CPUMatrix operator+(const CPUMatrix& o) const { return elementwise(o, std::plus<T>{}); }
    CPUMatrix operator-(const CPUMatrix& o) const { return elementwise(o, std::minus<T>{}); }
    CPUMatrix operator*(T s) const {
        return scaled([s](T x) { return x * s; });
    }
    CPUMatrix operator/(T s) const {
        return scaled([s](T x) { return x / s; });
    }

    CPUMatrix reshape(int r, int c) const {
        SOREI_CHECK(r * c == this->size());
        CPUMatrix out(Shape(r, c));
        std::copy(this->begin(), this->end(), out.begin());
        return out;
    }

    CPUMatrix transpose() const {
        CPUMatrix out(Shape(this->cols(), this->rows()));
        for (int r = 0; r < this->rows(); r++)
            for (int c = 0; c < this->cols(); c++)
                out(c, r) = (*this)(r, c);
        return out;
    }

    friend std::ostream& operator<<(std::ostream& os, const CPUMatrix& m) {
        os << "Matrix " << m.shape().str() << "\n" << std::fixed << std::setprecision(4);
        for (int r = 0; r < m.rows(); r++) {
            os << "  [ ";
            for (int c = 0; c < m.cols(); c++) {
                os << std::setw(10) << m(r, c);
                if (c + 1 < m.cols())
                    os << ", ";
            }
            os << " ]\n";
        }
        return os;
    }

  private:
    template <typename Op>
    CPUMatrix elementwise(const CPUMatrix& o, Op op) const {
        SOREI_CHECK(this->shape_ == o.shape_);
        CPUMatrix out(this->shape_);
        for (int i = 0; i < this->size(); i++)
            out(i) = op((*this)(i), o(i));
        return out;
    }

    template <typename Op>
    CPUMatrix scaled(Op op) const {
        CPUMatrix out(this->shape_);
        for (int i = 0; i < this->size(); i++)
            out(i) = op((*this)(i));
        return out;
    }
};

template <typename T>
using CPUPinnedMatrix = CPUMatrix<T, CPUPinnedArray<T>>;

template <typename T>
class GPUMatrix : public MatrixBase<T, GPUArray<T>> {
    using Base = MatrixBase<T, GPUArray<T>>;

  public:
    GPUMatrix() = default;

    explicit GPUMatrix(const Shape& shape)
        : Base(shape) {}

    // no-op if shape = shape()
    void resize(const Shape& shape) {
        if (shape == this->shape_)
            return;
        this->shape_ = shape;
        this->data_ = GPUArray<T>(shape.size());
        this->clear();
    }

    template <typename Src>
    void upload(const Src& src) {
        SOREI_CHECK(this->size() == src.size());
        SOREI_CHECK(this->bytes() == src.bytes());
        this->data_.upload(src);
    }

    template <typename Dst>
    void download(Dst& dst) const {
        SOREI_CHECK(this->size() == dst.size());
        SOREI_CHECK(this->bytes() == dst.bytes());
        this->data_.download(dst);
    }

    CPUMatrix<T> to_cpu() const {
        CPUMatrix<T> out(this->shape_);
        download(out);
        return out;
    }

    template <typename Src>
    static GPUMatrix from_cpu(const Src& src) {
        GPUMatrix out(src.shape());
        out.upload(src);
        return out;
    }
};

} // namespace sorei::tensor
