#pragma once

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <type_traits>

#include "shape.h"

namespace sorei::matrix {

template <typename T, typename Derived>
class HostMatrixBase {
  public:
    T& operator()(int r, int c) {
        SOREI_CHECK(r >= 0 && r < self().rows() && c >= 0 && c < self().cols());
        return self().data()[self().rows() * c + r];
    }

    const T& operator()(int r, int c) const {
        SOREI_CHECK(r >= 0 && r < self().rows() && c >= 0 && c < self().cols());
        return self().data()[self().rows() * c + r];
    }

    T& operator()(int i) {
        SOREI_CHECK(i >= 0 && i < self().size());
        return self().data()[i];
    }

    const T& operator()(int i) const {
        SOREI_CHECK(i >= 0 && i < self().size());
        return self().data()[i];
    }

    bool empty() const { return self().size() == 0; }

    T* begin() { return self().data(); }
    T* end() { return self().data() + self().size(); }
    const T* begin() const { return self().data(); }
    const T* end() const { return self().data() + self().size(); }

    void clear() { std::memset(self().data(), 0, self().bytes()); }
    void fill(const T& value) { std::fill_n(self().data(), self().size(), value); }

  private:
    Derived& self() { return static_cast<Derived&>(*this); }
    const Derived& self() const { return static_cast<const Derived&>(*this); }
};

} // namespace sorei::matrix
