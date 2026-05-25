#pragma once

#include <format>
#include <iostream>
#include <string>

#include "../util.h"

namespace sorei::matrix {

class Shape {
  public:
    Shape()
        : rows_(0),
          cols_(0) {}

    Shape(int rows, int cols)
        : rows_(rows),
          cols_(cols) {

        SOREI_CHECK(rows >= 0 && cols >= 0);
    }

    bool operator==(const Shape& other) const {
        return rows_ == other.rows_ && cols_ == other.cols_;
    }

    bool operator!=(const Shape& other) const { return !(*this == other); }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int size() const { return rows_ * cols_; }
    std::string str() const { return std::format("[{}x{}]", rows(), cols()); }

  private:
    int rows_, cols_;
};

inline std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << shape.str();
    return os;
}

} // namespace sorei::matrix
