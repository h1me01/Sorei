#pragma once

#include <cuda_runtime.h>

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

    void clear(cudaStream_t stream = 0) {
        if (data_)
            SOREI_CUDA_CHECK(cudaMemsetAsync(data_, 0, bytes(), stream));
    }

    template <typename HostSrc>
    void upload(const HostSrc& src) {
        SOREI_CHECK(src.shape() == shape_);
        SOREI_CUDA_CHECK(cudaMemcpy(data_, src.data(), bytes(), cudaMemcpyHostToDevice));
    }

    template <typename HostDst>
    void download(HostDst& dst) const {
        SOREI_CHECK(dst.shape() == shape_);
        SOREI_CUDA_CHECK(cudaMemcpy(dst.data(), data_, bytes(), cudaMemcpyDeviceToHost));
    }

    void upload(const DeviceMatrix& src) {
        SOREI_CHECK(src.shape() == shape_);
        SOREI_CUDA_CHECK(cudaMemcpy(data_, src.data_, bytes(), cudaMemcpyDeviceToDevice));
    }

    void download(DeviceMatrix& dst) const {
        SOREI_CHECK(dst.shape() == shape_);
        SOREI_CUDA_CHECK(cudaMemcpy(dst.data_, data_, bytes(), cudaMemcpyDeviceToDevice));
    }

    template <typename HostSrc>
    void upload_async(const HostSrc& src, cudaStream_t stream) {
        SOREI_CHECK(src.shape() == shape_);
        SOREI_CUDA_CHECK(
            cudaMemcpyAsync(data_, src.data(), bytes(), cudaMemcpyHostToDevice, stream)
        );
    }

    void upload_async(const DeviceMatrix& src, cudaStream_t stream) {
        SOREI_CHECK(src.shape() == shape_);
        SOREI_CUDA_CHECK(
            cudaMemcpyAsync(data_, src.data_, bytes(), cudaMemcpyDeviceToDevice, stream)
        );
    }

    template <typename HostDst>
    void download_async(HostDst& dst, cudaStream_t stream) const {
        SOREI_CHECK(dst.shape() == shape_);
        SOREI_CUDA_CHECK(
            cudaMemcpyAsync(dst.data(), data_, bytes(), cudaMemcpyDeviceToHost, stream)
        );
    }

    void download_async(DeviceMatrix& dst, cudaStream_t stream) const {
        SOREI_CHECK(dst.shape() == shape_);
        SOREI_CUDA_CHECK(
            cudaMemcpyAsync(dst.data_, data_, bytes(), cudaMemcpyDeviceToDevice, stream)
        );
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