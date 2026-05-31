#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <cuda_runtime.h>

#include "../cuda/utils.h"
#include "../util.h"
#include "shape.h"

namespace sorei::matrix {

template <typename T>
struct HeapPolicy {
    static T* allocate(size_t n) { return n > 0 ? new T[n] : nullptr; }
    static void deallocate(T* p, size_t) { delete[] p; }
};

template <typename T>
struct PinnedPolicy {
    static T* allocate(size_t n) {
        T* p = nullptr;
        if (n > 0)
            SOREI_CUDA_CHECK(cudaMallocHost(&p, n * sizeof(T)));
        return p;
    }
    static void deallocate(T* p, size_t) {
        if (p)
            SOREI_CUDA_CHECK(cudaFreeHost(p));
    }
};

template <typename T, template <typename> class Policy>
class HostMatrixImpl {
    using Pol = Policy<T>;

  public:
    HostMatrixImpl()
        : shape_(0, 0),
          data_(nullptr) {}

    explicit HostMatrixImpl(const Shape& shape)
        : shape_(shape),
          data_(Pol::allocate(shape.size())) {}

    HostMatrixImpl(int rows, int cols)
        : HostMatrixImpl(Shape{rows, cols}) {}

    HostMatrixImpl(const HostMatrixImpl& o)
        : shape_(o.shape_),
          data_(Pol::allocate(o.size())) {
        std::copy(o.data_, o.data_ + o.size(), data_);
    }

    HostMatrixImpl(HostMatrixImpl&& o) noexcept
        : shape_(o.shape_),
          data_(o.data_) {
        o.shape_ = Shape{0, 0};
        o.data_ = nullptr;
    }

    HostMatrixImpl& operator=(const HostMatrixImpl& o) {
        if (this != &o) {
            resize(o.shape_);
            std::copy(o.data_, o.data_ + o.size(), data_);
        }
        return *this;
    }

    HostMatrixImpl& operator=(HostMatrixImpl&& o) noexcept {
        if (this != &o) {
            Pol::deallocate(data_, size());
            shape_ = o.shape_;
            data_ = o.data_;
            o.shape_ = Shape{0, 0};
            o.data_ = nullptr;
        }
        return *this;
    }

    ~HostMatrixImpl() { Pol::deallocate(data_, size()); }

    int rows() const { return shape_.rows(); }
    int cols() const { return shape_.cols(); }
    int size() const { return shape_.size(); }
    Shape shape() const { return shape_; }
    size_t bytes() const { return static_cast<size_t>(size()) * sizeof(T); }
    T* data() { return data_; }
    const T* data() const { return data_; }

    bool empty() const { return size() == 0; }

    T& operator()(int r, int c) {
        SOREI_CHECK(r >= 0 && r < rows() && c >= 0 && c < cols());
        return data_[rows() * c + r];
    }
    const T& operator()(int r, int c) const {
        SOREI_CHECK(r >= 0 && r < rows() && c >= 0 && c < cols());
        return data_[rows() * c + r];
    }
    T& operator()(int i) {
        SOREI_CHECK(i >= 0 && i < size());
        return data_[i];
    }
    const T& operator()(int i) const {
        SOREI_CHECK(i >= 0 && i < size());
        return data_[i];
    }

    T* begin() { return data_; }
    T* end() { return data_ + size(); }
    const T* begin() const { return data_; }
    const T* end() const { return data_ + size(); }

    void clear() { std::memset(data_, 0, bytes()); }
    void fill(const T& val) { std::fill_n(data_, size(), val); }

    void resize(const Shape& new_shape) {
        if (new_shape == shape_)
            return;
        Pol::deallocate(data_, size());
        shape_ = new_shape;
        data_ = Pol::allocate(new_shape.size());
    }

    HostMatrixImpl transpose() const {
        HostMatrixImpl out(cols(), rows());
        for (int r = 0; r < rows(); ++r)
            for (int c = 0; c < cols(); ++c)
                out(c, r) = (*this)(r, c);
        return out;
    }

  private:
    Shape shape_;
    T* data_;
};

template <typename T>
using HostMatrix = HostMatrixImpl<T, HeapPolicy>;

template <typename T>
using HostPinnedMatrix = HostMatrixImpl<T, PinnedPolicy>;

} // namespace sorei::matrix
