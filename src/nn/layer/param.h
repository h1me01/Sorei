#pragma once

#include <cmath>
#include <limits>
#include <random>

#include "../../rng/rng.h"
#include "layer.h"

namespace sorei::nn::layer {

class Param : public TypedLayer<float> {
  public:
    Param(const tensor::Shape& shape, const std::string& name = "Param")
        : TypedLayer<float>(name),
          shape_(shape) {}

    void uniform_init(float min_val, float max_val) {
        CHECK(min_val <= max_val);

        tensor::CPUMatrix<float> result(shape_);
        for (int i = 0; i < result.size(); i++) {
            result(i) = std::uniform_real_distribution<float>(min_val, max_val)(
                rng::get_thread_local_rng()
            );
        }
        data().upload(result);

        grad().clear();
    }

    void he_init() {
        tensor::CPUMatrix<float> result(shape_);
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
    int input_dim() const { return shape_.cols(); }
    int output_dim() const { return shape_.rows(); }
    float lower_bound() const { return lower_bound_; }
    float upper_bound() const { return upper_bound_; }
    tensor::Shape shape() const override { return shape_; }

  private:
    tensor::Shape shape_;

    float lower_bound_ = std::numeric_limits<float>::lowest();
    float upper_bound_ = std::numeric_limits<float>::max();
};

} // namespace sorei::nn::layer
