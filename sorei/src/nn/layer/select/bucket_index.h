#pragma once

#include "../input.h"
#include "../layer.h"

namespace sorei::nn {

class BucketIndex : public Input<int> {
  public:
    BucketIndex(const std::string& name, int count, int size)
        : Input<int>(name, matrix::Shape(1, size)),
          count_(count) {

        SOREI_CHECK(count > 0);
    }

    int count() const { return count_; }

  private:
    int count_;
};

} // namespace sorei::nn
