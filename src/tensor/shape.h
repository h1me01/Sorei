#pragma once

#include <iostream>
#include <string>

#include "../misc.h"

namespace sorei::tensor {

class Shape {
  public:
    Shape(int rows, int cols)
        : rows_(rows),
          cols_(cols) {

        CHECK(rows >= 0 && cols >= 0);
    }

    bool operator==(const Shape& other) const {
        return rows_ == other.rows_ && cols_ == other.cols_;
    }

    bool operator!=(const Shape& other) const { return !(*this == other); }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int size() const { return rows_ * cols_; }

    std::string str() const {
        return "(" + std::to_string(rows()) + " x " + std::to_string(cols()) + ")";
    }

  private:
    int rows_, cols_;
};

inline std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << shape.str();
    return os;
}

} // namespace sorei::tensor
