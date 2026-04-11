#pragma once

#include <cmath>
#include <limits>
#include <random>

#include "../../rng/rng.h"
#include "layer.h"

namespace nn::layer {

class Param : public TypedLayer<float> {
  public:
    Param(const data::Shape& shape, const std::string& name = "Param")
        : TypedLayer<float>(name),
          shape_(shape) {}

    void uniform_init(float min_val, float max_val) {
        CHECK(min_val <= max_val);

        data::CPUMatrix<float> result(shape_);
        for (int i = 0; i < result.size(); i++) {
            result(i) = std::uniform_real_distribution<float>(min_val, max_val)(
                rng::get_thread_local_rng()
            );
        }
        data().upload(result);

        grad().clear();
    }

    void he_init() {
        data::CPUMatrix<float> result(shape_);
        for (int i = 0; i < result.size(); i++) {
            result(i) = std::normal_distribution<float>(0.0, std::sqrt(2.0 / result.cols()))(
                rng::get_thread_local_rng()
            );
        }
        data().upload(result);

        grad().clear();
    }

    void set_bounds(float min_val, float max_val) {
        CHECK(min_val <= max_val);
        lower_bound_ = min_val;
        upper_bound_ = max_val;
    }

    bool requires_grad() const override { return true; }
    float lower_bound() const { return lower_bound_; }
    float upper_bound() const { return upper_bound_; }
    data::Shape shape() const override { return shape_; }

  private:
    data::Shape shape_;

    float lower_bound_ = std::numeric_limits<float>::lowest();
    float upper_bound_ = std::numeric_limits<float>::max();
};

} // namespace nn::layer
