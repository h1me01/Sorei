#pragma once

#include <vector>

#include "../tensor/include.h"

namespace sorei::nn {

// Column-major 1D or 2D tensor
template <typename T>
class Tensor {
  public:
    enum StorageType : unsigned {
        CPU = 1 << 0,
        CPU_PINNED = 1 << 1,
        GPU = 1 << 2,
    };

  public:
    Tensor() = default;

    explicit Tensor(std::vector<int> shape, StorageType storage = CPU)
        : shape_(std::move(shape)),
          storage_(storage) {

        if (shape_.size() > 2)
            error("Tensor: only 1D or 2D shapes supported");
        for (int d : shape_)
            if (d <= 0)
                error("Tensor: all dimensions must be > 0");

        if (has_flag(storage_, CPU) && has_flag(storage_, CPU_PINNED))
            error("Tensor: CPU and CPU_PINNED are mutually exclusive");

        allocate();
    }

    T operator[](int i) const { return host_ref(flat1d(i)); }
    T& operator[](int i) { return host_ref(flat1d(i)); }

    T operator()(int r, int c) const { return host_ref(flat2d(r, c)); }
    T& operator()(int r, int c) { return host_ref(flat2d(r, c)); }

    int size() const {
        if (shape_.empty())
            return 0;
        return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>{});
    }

    bool empty() const { return size() == 0; }
    std::vector<int> shape() const { return shape_; }

    bool has_cpu_data() const { return !cpu_data_.empty(); }
    bool has_cpu_pinned_data() const { return !cpu_pinned_data_.empty(); }
    bool has_gpu_data() const { return !gpu_data_.empty(); }

    // fills only host storage
    void fill(T value) {
        cpu_data_.fill(value);
        cpu_pinned_data_.fill(value);
    }

    void resize(std::vector<int> shape) {
        if (shape.size() > 2)
            throw std::invalid_argument("Tensor: only 1D or 2D shapes supported");
        shape_ = std::move(shape);
        allocate();
    }

    void host_to_device() {
        if (has_cpu_data())
            gpu_data_.upload(cpu_data_);
        else if (has_cpu_pinned_data())
            gpu_data_.upload(cpu_pinned_data_);
    }

    void device_to_host() {
        if (has_cpu_data())
            gpu_data_.download(cpu_data_);
        else if (has_cpu_pinned_data())
            gpu_data_.download(cpu_pinned_data_);
    }

    tensor::CPUArray<T>& cpu_data() { return cpu_data_; }
    const tensor::CPUArray<T>& cpu_data() const { return cpu_data_; }
    tensor::CPUPinnedArray<T>& cpu_pinned_data() { return cpu_pinned_data_; }
    const tensor::CPUPinnedArray<T>& cpu_pinned_data() const { return cpu_pinned_data_; }
    tensor::GPUArray<T>& gpu_data() { return gpu_data_; }
    const tensor::GPUArray<T>& gpu_data() const { return gpu_data_; }

  private:
    void allocate() {
        int n = size();
        if (has_flag(storage_, CPU))
            cpu_data_.resize(n);
        if (has_flag(storage_, CPU_PINNED))
            cpu_pinned_data_.resize(n);
        if (has_flag(storage_, GPU))
            gpu_data_.resize(n);
    }

    int flat1d(int i) const {
        SOREI_CHECK(shape_.size() == 1);
        SOREI_CHECK(i >= 0 && i < shape_[0]);
        return i;
    }

    int flat2d(int r, int c) const {
        SOREI_CHECK(shape_.size() == 2);
        SOREI_CHECK(r >= 0 && r < shape_[0]);
        SOREI_CHECK(c >= 0 && c < shape_[1]);
        return r + c * shape_[0];
    }

    static bool has_flag(StorageType flags, StorageType bit) { return (flags & bit) != 0; }

    T& host_ref(int idx) { return has_cpu_data() ? cpu_data_[idx] : cpu_pinned_data_[idx]; }

    const T& host_ref(int idx) const {
        return has_cpu_data() ? cpu_data_[idx] : cpu_pinned_data_[idx];
    }

    std::vector<int> shape_;
    StorageType storage_{CPU};
    tensor::CPUArray<T> cpu_data_;
    tensor::CPUPinnedArray<T> cpu_pinned_data_;
    tensor::GPUArray<T> gpu_data_;
};

} // namespace sorei::nn