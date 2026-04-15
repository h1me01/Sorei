#pragma once

#include <cmath>
#include <format>
#include <string>

#include "lr_scheduler.h"

namespace sorei::nn::lr_sched {

class StepLR : public LRScheduler {
  public:
    StepLR(float lr, float gamma, int step_size)
        : LRScheduler(lr),
          base_lr_(lr),
          gamma_(gamma),
          step_size_(step_size) {

        SOREI_CHECK(step_size > 0);
    }

    float base_lr() const { return base_lr_; }
    float gamma() const { return gamma_; }
    int step_size() const { return step_size_; }

    std::string info() const override {
        return std::format(
            "StepLR(lr={:.6g}, gamma={:.6g}, step_size={})", base_lr_, gamma_, step_size_
        );
    }

  private:
    float base_lr_;
    float gamma_;
    int step_size_;

    float lr_update(int step) override { return base_lr_ * std::pow(gamma_, step / step_size_); }
};

} // namespace sorei::nn::lr_sched
