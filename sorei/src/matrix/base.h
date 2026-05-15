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

namespace {

// column-major flat index
inline int cm_index(int rows, int r, int c) { return rows * c + r; }

} // namespace

template <typename T, typename Derived>
class HostMatrixBase {
  protected:
    T& at(int i) {
        SOREI_CHECK(i >= 0 && i < self().size());
        return self().data()[i];
    }
    const T& at(int i) const {
        SOREI_CHECK(i >= 0 && i < self().size());
        return self().data()[i];
    }

  public:
    bool in_bounds(int r, int c) const {
        return r >= 0 && r < self().rows() && c >= 0 && c < self().cols();
    }

    T& operator()(int r, int c) {
        SOREI_CHECK(in_bounds(r, c));
        return at(cm_index(self().rows(), r, c));
    }
    const T& operator()(int r, int c) const {
        SOREI_CHECK(in_bounds(r, c));
        return at(cm_index(self().rows(), r, c));
    }

    T& operator()(int i) { return at(i); }
    const T& operator()(int i) const { return at(i); }

    bool empty() const { return self().size() == 0; }

    T* begin() { return self().data(); }
    T* end() { return self().data() + self().size(); }
    const T* begin() const { return self().data(); }
    const T* end() const { return self().data() + self().size(); }
    const T* cbegin() const { return self().data(); }
    const T* cend() const { return self().data() + self().size(); }

    void clear() { std::memset(self().data(), 0, self().bytes()); }
    void fill(const T& value) { std::fill_n(self().data(), self().size(), value); }

    Derived operator+(const Derived& o) const { return elementwise(o, std::plus<T>{}); }
    Derived operator-(const Derived& o) const { return elementwise(o, std::minus<T>{}); }
    Derived operator*(T s) const {
        return scaled([s](T x) { return x * s; });
    }
    Derived operator/(T s) const {
        return scaled([s](T x) { return x / s; });
    }

    Derived& operator+=(const Derived& o) { return inplace(o, std::plus<T>{}); }
    Derived& operator-=(const Derived& o) { return inplace(o, std::minus<T>{}); }
    Derived& operator*=(T s) {
        for (int i = 0; i < self().size(); ++i)
            self().data()[i] *= s;
        return self();
    }
    Derived& operator/=(T s) {
        for (int i = 0; i < self().size(); ++i)
            self().data()[i] /= s;
        return self();
    }

    Derived reshape(const Shape& shape) const {
        SOREI_CHECK(shape.size() == self().size());
        Derived out(shape);
        std::copy(cbegin(), cend(), out.begin());
        return out;
    }

    Derived transpose() const {
        Derived out(Shape{self().cols(), self().rows()});
        for (int r = 0; r < self().rows(); ++r)
            for (int c = 0; c < self().cols(); ++c)
                out(c, r) = (*this)(r, c);
        return out;
    }

    friend std::ostream& operator<<(std::ostream& os, const HostMatrixBase& m) {
        const auto& s = m.self();
        os << "Matrix " << s.shape() << "\n" << std::fixed << std::setprecision(4);
        for (int r = 0; r < s.rows(); ++r) {
            os << "  [ ";
            for (int c = 0; c < s.cols(); ++c) {
                os << std::setw(10) << s(r, c);
                if (c + 1 < s.cols())
                    os << ", ";
            }
            os << " ]\n";
        }
        return os;
    }

  private:
    Derived& self() { return static_cast<Derived&>(*this); }
    const Derived& self() const { return static_cast<const Derived&>(*this); }

    template <typename Op>
    Derived elementwise(const Derived& o, Op op) const {
        SOREI_CHECK(self().shape() == o.shape());
        Derived out(self().shape());
        const T* a = self().data();
        const T* b = o.data();
        T* d = out.data();
        for (int i = 0, n = self().size(); i < n; ++i)
            d[i] = op(a[i], b[i]);
        return out;
    }

    template <typename Op>
    Derived scaled(Op op) const {
        Derived out(self().shape());
        const T* a = self().data();
        T* d = out.data();
        for (int i = 0, n = self().size(); i < n; ++i)
            d[i] = op(a[i]);
        return out;
    }

    template <typename Op>
    Derived& inplace(const Derived& o, Op op) {
        SOREI_CHECK(self().shape() == o.shape());
        T* a = self().data();
        const T* b = o.data();
        for (int i = 0, n = self().size(); i < n; ++i)
            a[i] = op(a[i], b[i]);
        return self();
    }
};

} // namespace sorei::matrix
