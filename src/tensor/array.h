#pragma once

#include <cstring>
#include <memory>

#include "../misc.h"

namespace sorei::tensor {

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
        : size_(size) {
        if (size > 0)
            SOREI_CUDA_CHECK(cudaMallocHost(&ptr_, size * sizeof(T)));
        clear();
    }

    ~PinnedCPUArray() {
        if (ptr_)
            cudaFreeHost(ptr_);
    }

    PinnedCPUArray(const PinnedCPUArray& other)
        : size_(other.size_) {
        if (size_ > 0) {
            SOREI_CUDA_CHECK(cudaMallocHost(&ptr_, bytes()));
            std::memcpy(ptr_, other.ptr_, bytes());
        }
    }

    PinnedCPUArray& operator=(const PinnedCPUArray& other) {
        if (this != &other) {
            PinnedCPUArray tmp(other);
            *this = std::move(tmp);
        }
        return *this;
    }

    PinnedCPUArray(PinnedCPUArray&& other) noexcept
        : size_(other.size_),
          ptr_(other.ptr_) {
        other.size_ = 0;
        other.ptr_ = nullptr;
    }

    PinnedCPUArray& operator=(PinnedCPUArray&& other) noexcept {
        if (this != &other) {
            if (ptr_)
                cudaFreeHost(ptr_);
            size_ = other.size_;
            ptr_ = other.ptr_;
            other.size_ = 0;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    int size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }
    bool empty() const { return size_ == 0; }
    T* data() const { return ptr_; }

    T operator[](int i) const {
        SOREI_CHECK(i >= 0 && i < size_);
        return ptr_[i];
    }

    T& operator[](int i) {
        SOREI_CHECK(i >= 0 && i < size_);
        return ptr_[i];
    }

    void clear() { std::memset(ptr_, 0, bytes()); }
    void fill(T value) { std::fill_n(ptr_, size_, value); }

    void resize(int size) {
        if (size == size_)
            return;
        if (ptr_)
            cudaFreeHost(ptr_);
        ptr_ = nullptr;
        size_ = size;
        if (size > 0)
            SOREI_CUDA_CHECK(cudaMallocHost(&ptr_, bytes()));
        clear();
    }

  private:
    int size_ = 0;
    T* ptr_ = nullptr;
};

template <typename T>
class GPUArray {
  public:
    GPUArray() = default;

    explicit GPUArray(int size)
        : size_(size) {
        if (size > 0)
            SOREI_CUDA_CHECK(cudaMalloc(&ptr_, size * sizeof(T)));
        clear();
    }

    ~GPUArray() {
        if (ptr_)
            cudaFree(ptr_);
    }

    GPUArray(const GPUArray& other)
        : size_(other.size_) {
        if (size_ > 0) {
            SOREI_CUDA_CHECK(cudaMalloc(&ptr_, bytes()));
            SOREI_CUDA_CHECK(cudaMemcpy(ptr_, other.ptr_, bytes(), cudaMemcpyDeviceToDevice));
        }
    }

    GPUArray& operator=(const GPUArray& other) {
        if (this != &other) {
            GPUArray tmp(other);
            *this = std::move(tmp);
        }
        return *this;
    }

    GPUArray(GPUArray&& other) noexcept
        : size_(other.size_),
          ptr_(other.ptr_) {
        other.size_ = 0;
        other.ptr_ = nullptr;
    }

    GPUArray& operator=(GPUArray&& other) noexcept {
        if (this != &other) {
            if (ptr_)
                cudaFree(ptr_);
            size_ = other.size_;
            ptr_ = other.ptr_;
            other.size_ = 0;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    int size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }
    bool empty() const { return size_ == 0; }
    T* data() const { return ptr_; }

    void clear() { SOREI_CUDA_CHECK(cudaMemsetAsync(ptr_, 0, bytes(), 0)); }

    void resize(int size) {
        if (size == size_)
            return;
        if (ptr_)
            cudaFree(ptr_);
        ptr_ = nullptr;
        size_ = size;
        if (size > 0)
            SOREI_CUDA_CHECK(cudaMalloc(&ptr_, bytes()));
        clear();
    }

    template <typename Src>
    void upload(const Src& src) {
        SOREI_CHECK(src.size() == size_);
        SOREI_CHECK(src.bytes() == bytes());
        SOREI_CUDA_CHECK(cudaMemcpy(ptr_, src.data(), bytes(), cudaMemcpyHostToDevice));
    }

    template <typename Dst>
    void download(Dst& dst) const {
        SOREI_CHECK(size_ == dst.size());
        SOREI_CHECK(bytes() == dst.bytes());
        SOREI_CUDA_CHECK(cudaMemcpy(dst.data(), ptr_, bytes(), cudaMemcpyDeviceToHost));
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
    T* ptr_ = nullptr;
};

} // namespace sorei::tensor
