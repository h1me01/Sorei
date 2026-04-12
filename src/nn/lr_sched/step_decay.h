#pragma once

#include <cmath>
#include <format>
#include <string>

#include "lr_scheduler.h"

namespace sorei::nn::lr_sched {

class StepDecay : public LRScheduler {
  public:
    StepDecay(float lr, float gamma, int step_size)
        : LRScheduler(lr),
          base_lr_(lr),
          gamma_(gamma),
          step_size_(step_size) {

        SOREI_CHECK(step_size > 0);
    }

    std::string info() const override {
        return std::format(
            "StepDecay(lr={:.6g}, gamma={:.6g}, step_size={})", base_lr_, gamma_, step_size_
        );
    }

  private:
    float base_lr_, gamma_;
    int step_size_;

    float lr_update(int step) override { return base_lr_ * std::pow(gamma_, step / step_size_); }
};

} // namespace sorei::nn::lr_sched
