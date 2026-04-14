#pragma once

#include <vector>

#include "../tensor/include.h"

namespace sorei::nn {

// Column-major 1D or 2D tensor
template <typename T>
class Tensor {
  public:
    enum StorageType : unsigned {
        HOST = 1 << 0,
        HOST_PINNED = 1 << 1,
        DEVICE = 1 << 2,
    };

  public:
    Tensor() = default;

    explicit Tensor(std::vector<int> shape, StorageType storage = HOST)
        : shape_(std::move(shape)),
          storage_(storage) {

        if (shape_.size() > 2)
            error("Tensor: only 1D or 2D shapes supported");
        for (int d : shape_)
            if (d <= 0)
                error("Tensor: all dimensions must be > 0");

        if (has_flag(storage_, HOST) && has_flag(storage_, HOST_PINNED))
            error("Tensor: HOST and HOST_PINNED are mutually exclusive");

        allocate();
    }

    T operator[](int i) const {
        return has_host_data() ? host_data_[flat1d(i)] : host_pinned_data_[flat1d(i)];
    }

    T& operator[](int i) {
        return has_host_data() ? host_data_[flat1d(i)] : host_pinned_data_[flat1d(i)];
    }

    T operator()(int r, int c) const {
        return has_host_data() ? host_data_[flat2d(r, c)] : host_pinned_data_[flat2d(r, c)];
    }

    T& operator()(int r, int c) {
        return has_host_data() ? host_data_[flat2d(r, c)] : host_pinned_data_[flat2d(r, c)];
    }

    int size() const {
        if (shape_.empty())
            return 0;
        return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>{});
    }

    bool empty() const { return size() == 0; }
    std::vector<int> shape() const { return shape_; }

    bool has_host_data() const { return !host_data_.empty(); }
    bool has_host_pinned_data() const { return !host_pinned_data_.empty(); }
    bool has_device_data() const { return !device_data_.empty(); }

    // fills only host storage
    void fill(T value) {
        host_data_.fill(value);
        host_pinned_data_.fill(value);
    }

    void resize(std::vector<int> shape) {
        if (shape.size() > 2)
            throw std::invalid_argument("Tensor: only 1D or 2D shapes supported");
        shape_ = std::move(shape);
        allocate();
    }

    void host_to_device() {
        if (has_host_data())
            device_data_.upload(host_data_);
        else if (has_host_pinned_data())
            device_data_.upload(host_pinned_data_);
    }

    void device_to_host() {
        if (has_host_data())
            device_data_.download(host_data_);
        else if (has_host_pinned_data())
            device_data_.download(host_pinned_data_);
    }

    tensor::HostArray<T>& host_data() { return host_data_; }
    const tensor::HostArray<T>& host_data() const { return host_data_; }
    tensor::HostPinnedArray<T>& host_pinned_data() { return host_pinned_data_; }
    const tensor::HostPinnedArray<T>& host_pinned_data() const { return host_pinned_data_; }
    tensor::DeviceArray<T>& device_data() { return device_data_; }
    const tensor::DeviceArray<T>& device_data() const { return device_data_; }

  private:
    void allocate() {
        int n = size();
        if (has_flag(storage_, HOST))
            host_data_.resize(n);
        if (has_flag(storage_, HOST_PINNED))
            host_pinned_data_.resize(n);
        if (has_flag(storage_, DEVICE))
            device_data_.resize(n);
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

    std::vector<int> shape_;
    StorageType storage_{HOST};
    tensor::HostArray<T> host_data_;
    tensor::HostPinnedArray<T> host_pinned_data_;
    tensor::DeviceArray<T> device_data_;
};

} // namespace sorei::nn