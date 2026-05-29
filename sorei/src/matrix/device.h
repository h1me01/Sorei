#pragma once

#include <cuda_runtime.h>

#include "../cuda/utils.h"
#include "host.h"
#include "host_pinned.h"

namespace sorei::matrix {

template <typename T>
class DeviceMatrix {
  public:
    DeviceMatrix()
        : shape_(0, 0),
          data_(nullptr) {}

    explicit DeviceMatrix(const Shape& shape)
        : shape_(shape),
          data_(nullptr) {
        alloc();
    }

    DeviceMatrix(int rows, int cols)
        : DeviceMatrix(Shape{rows, cols}) {}

    DeviceMatrix(const DeviceMatrix& o)
        : shape_(o.shape_),
          data_(nullptr) {
        alloc();
        if (size() > 0)
            SOREI_CUDA_CHECK(cudaMemcpy(data_, o.data_, bytes(), cudaMemcpyDeviceToDevice));
    }

    DeviceMatrix(DeviceMatrix&& o) noexcept
        : shape_(o.shape_),
          data_(o.data_) {
        o.shape_ = Shape{0, 0};
        o.data_ = nullptr;
    }

    DeviceMatrix& operator=(const DeviceMatrix& o) {
        if (this != &o) {
            if (shape_ != o.shape_) {
                free();
                shape_ = o.shape_;
                alloc();
            }
            if (size() > 0)
                SOREI_CUDA_CHECK(cudaMemcpy(data_, o.data_, bytes(), cudaMemcpyDeviceToDevice));
        }
        return *this;
    }

    DeviceMatrix& operator=(DeviceMatrix&& o) noexcept {
        if (this != &o) {
            free();
            shape_ = o.shape_;
            data_ = o.data_;
            o.shape_ = Shape{0, 0};
            o.data_ = nullptr;
        }
        return *this;
    }

    ~DeviceMatrix() { free(); }

    int rows() const { return shape_.rows(); }
    int cols() const { return shape_.cols(); }
    int size() const { return shape_.size(); }
    Shape shape() const { return shape_; }
    bool empty() const { return size() == 0; }
    size_t bytes() const { return static_cast<size_t>(size()) * sizeof(T); }

    T* data() const { return data_; }

    void resize(const Shape& new_shape) {
        if (new_shape == shape_)
            return;
        free();
        shape_ = new_shape;
        alloc();
    }

    void clear() {
        if (data_)
            SOREI_CUDA_CHECK(cudaMemset(data_, 0, bytes()));
    }

    template <typename Src>
    void upload(const Src& src) {
        SOREI_CHECK(src.shape() == shape_);
        if constexpr (std::is_same_v<Src, DeviceMatrix>) {
            SOREI_CUDA_CHECK(cudaMemcpy(data_, src.data(), bytes(), cudaMemcpyDeviceToDevice));
        } else {
            SOREI_CUDA_CHECK(cudaMemcpy(data_, src.data(), bytes(), cudaMemcpyHostToDevice));
        }
    }

    template <typename Dst>
    void download(Dst& dst) const {
        SOREI_CHECK(dst.shape() == shape_);
        if constexpr (std::is_same_v<Dst, DeviceMatrix>) {
            SOREI_CUDA_CHECK(cudaMemcpy(dst.data(), data_, bytes(), cudaMemcpyDeviceToDevice));
        } else {
            SOREI_CUDA_CHECK(cudaMemcpy(dst.data(), data_, bytes(), cudaMemcpyDeviceToHost));
        }
    }

    template <typename Src>
    void upload_async(const Src& src, cudaStream_t stream = 0) {
        SOREI_CHECK(src.shape() == shape_);
        if constexpr (std::is_same_v<Src, DeviceMatrix>) {
            SOREI_CUDA_CHECK(
                cudaMemcpyAsync(data_, src.data(), bytes(), cudaMemcpyDeviceToDevice, stream)
            );
        } else {
            SOREI_CUDA_CHECK(
                cudaMemcpyAsync(data_, src.data(), bytes(), cudaMemcpyHostToDevice, stream)
            );
        }
    }

    template <typename Dst>
    void download_async(Dst& dst, cudaStream_t stream = 0) const {
        SOREI_CHECK(dst.shape() == shape_);
        if constexpr (std::is_same_v<Dst, DeviceMatrix>) {
            SOREI_CUDA_CHECK(
                cudaMemcpyAsync(dst.data(), data_, bytes(), cudaMemcpyDeviceToDevice, stream)
            );
        } else {
            SOREI_CUDA_CHECK(
                cudaMemcpyAsync(dst.data(), data_, bytes(), cudaMemcpyDeviceToHost, stream)
            );
        }
    }

    HostMatrix<T> to_host() const {
        HostMatrix<T> out(shape_);
        download(out);
        return out;
    }

    HostPinnedMatrix<T> to_pinned() const {
        HostPinnedMatrix<T> out(shape_);
        download(out);
        return out;
    }

    template <typename HostSrc>
    static DeviceMatrix from_host(const HostSrc& src) {
        DeviceMatrix buf(src.shape());
        buf.upload(src);
        return buf;
    }

  private:
    Shape shape_;
    T* data_;

    void alloc() {
        if (size() > 0)
            SOREI_CUDA_CHECK(cudaMalloc(&data_, bytes()));
    }

    void free() {
        if (data_) {
            SOREI_CUDA_CHECK(cudaFree(data_));
            data_ = nullptr;
        }
    }
};

} // namespace sorei::matrix