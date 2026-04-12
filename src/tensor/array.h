#pragma once

#include <cstring>
#include <memory>

#include "../misc.h"

namespace sorei::tensor {

template <typename T>
class CudaDevicePtr {
  public:
    CudaDevicePtr() = default;

    explicit CudaDevicePtr(size_t count) {
        if (count > 0)
            SOREI_CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }

    ~CudaDevicePtr() {
        if (ptr_)
            cudaFree(ptr_);
    }

    CudaDevicePtr(const CudaDevicePtr&) = delete;
    CudaDevicePtr& operator=(const CudaDevicePtr&) = delete;

    CudaDevicePtr(CudaDevicePtr&& other) noexcept
        : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    CudaDevicePtr& operator=(CudaDevicePtr&& other) noexcept {
        if (this != &other) {
            if (ptr_)
                cudaFree(ptr_);

            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    T* get() const { return ptr_; }

    explicit operator bool() const { return ptr_ != nullptr; }

  private:
    T* ptr_ = nullptr;
};

template <typename T>
class CudaHostPtr {
  public:
    CudaHostPtr() = default;

    explicit CudaHostPtr(size_t count) {
        if (count > 0)
            SOREI_CUDA_CHECK(cudaMallocHost(&ptr_, count * sizeof(T)));
    }

    ~CudaHostPtr() {
        if (ptr_)
            cudaFreeHost(ptr_);
    }

    CudaHostPtr(const CudaHostPtr&) = delete;
    CudaHostPtr& operator=(const CudaHostPtr&) = delete;

    CudaHostPtr(CudaHostPtr&& other) noexcept
        : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    CudaHostPtr& operator=(CudaHostPtr&& other) noexcept {
        if (this != &other) {
            if (ptr_)
                cudaFreeHost(ptr_);

            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    T* get() const { return ptr_; }

    explicit operator bool() const { return ptr_ != nullptr; }

  private:
    T* ptr_ = nullptr;
};

template <typename T>
class CPUArray {
  public:
    CPUArray() = default;

    explicit CPUArray(int size)
        : size_(size),
          data_(std::make_unique<T[]>(size)) {
        clear();
    }

    CPUArray(const CPUArray& other)
        : size_(other.size_),
          data_(std::make_unique<T[]>(other.size_)) {
        std::memcpy(data_.get(), other.data_.get(), bytes());
    }

    CPUArray& operator=(const CPUArray& other) {
        if (this != &other) {
            CPUArray tmp(other);
            *this = std::move(tmp);
        }
        return *this;
    }

    CPUArray(CPUArray&&) noexcept = default;
    CPUArray& operator=(CPUArray&&) noexcept = default;

    int size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }
    bool empty() const { return size_ == 0; }
    T* data() const { return data_.get(); }

    T operator[](int i) const {
        SOREI_CHECK(i >= 0 && i < size_);
        return data_.get()[i];
    }

    T& operator[](int i) {
        SOREI_CHECK(i >= 0 && i < size_);
        return data_.get()[i];
    }

    void clear() { std::memset(data_.get(), 0, bytes()); }
    void fill(T value) { std::fill_n(data_.get(), size_, value); }

    // no-op if size = size()
    void resize(int size) {
        if (size == size_)
            return;
        size_ = size;
        data_ = std::make_unique<T[]>(size);
        clear();
    }

  private:
    int size_ = 0;
    std::unique_ptr<T[]> data_;
};

template <typename T>
class PinnedCPUArray {
  public:
    PinnedCPUArray() = default;

    explicit PinnedCPUArray(int size)
        : size_(size),
          data_(size) {
        clear();
    }

    PinnedCPUArray(const PinnedCPUArray& other)
        : size_(other.size_),
          data_(other.size_) {
        std::memcpy(data_.get(), other.data_.get(), bytes());
    }

    PinnedCPUArray& operator=(const PinnedCPUArray& other) {
        if (this != &other) {
            PinnedCPUArray tmp(other);
            *this = std::move(tmp);
        }
        return *this;
    }

    PinnedCPUArray(PinnedCPUArray&&) noexcept = default;
    PinnedCPUArray& operator=(PinnedCPUArray&&) noexcept = default;

    int size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }
    bool empty() const { return size_ == 0; }
    T* data() const { return data_.get(); }

    T operator[](int i) const {
        SOREI_CHECK(i >= 0 && i < size_);
        return data_.get()[i];
    }

    T& operator[](int i) {
        SOREI_CHECK(i >= 0 && i < size_);
        return data_.get()[i];
    }

    void clear() { std::memset(data_.get(), 0, bytes()); }
    void fill(T value) { std::fill_n(data_.get(), size_, value); }

    // no-op if size = size()
    void resize(int size) {
        if (size == size_)
            return;
        size_ = size;
        data_ = CudaHostPtr<T>(size);
        clear();
    }

  private:
    int size_ = 0;
    CudaHostPtr<T> data_;
};

template <typename T>
class GPUArray {
  public:
    GPUArray() = default;

    explicit GPUArray(int size)
        : size_(size),
          data_(size) {
        clear();
    }

    GPUArray(const GPUArray& other)
        : size_(other.size_),
          data_(other.size_) {
        SOREI_CUDA_CHECK(
            cudaMemcpy(data_.get(), other.data_.get(), bytes(), cudaMemcpyDeviceToDevice)
        );
    }

    GPUArray& operator=(const GPUArray& other) {
        if (this != &other) {
            GPUArray tmp(other);
            *this = std::move(tmp);
        }
        return *this;
    }

    GPUArray(GPUArray&&) noexcept = default;
    GPUArray& operator=(GPUArray&&) noexcept = default;

    int size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }
    bool empty() const { return size_ == 0; }
    T* data() const { return data_.get(); }

    void clear() { SOREI_CUDA_CHECK(cudaMemsetAsync(data_.get(), 0, bytes(), 0)); }

    // no-op if size = size()
    void resize(int size) {
        if (size == size_)
            return;
        size_ = size;
        data_ = CudaDevicePtr<T>(size);
        clear();
    }

    template <typename Src>
    void upload(const Src& src) {
        SOREI_CHECK(src.size() == size_);
        SOREI_CHECK(src.bytes() == bytes());
        SOREI_CUDA_CHECK(cudaMemcpy(data_.get(), src.data(), bytes(), cudaMemcpyHostToDevice));
    }

    template <typename Dst>
    void download(Dst& dst) const {
        SOREI_CHECK(size_ == dst.size());
        SOREI_CHECK(bytes() == dst.bytes());
        SOREI_CUDA_CHECK(cudaMemcpy(dst.data(), data_.get(), bytes(), cudaMemcpyDeviceToHost));
    }

    CPUArray<T> to_cpu() const {
        CPUArray<T> out(size_);
        download(out);
        return out;
    }

    template <typename Src>
    static GPUArray from_cpu(const Src& src) {
        GPUArray buf(src.size());
        buf.upload(src);
        return buf;
    }

  private:
    int size_ = 0;
    CudaDevicePtr<T> data_;
};

} // namespace sorei::tensor
