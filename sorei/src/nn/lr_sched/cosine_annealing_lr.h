#pragma once

#include <cmath>
#include <format>
#include <numbers>
#include <string>

#include "lr_scheduler.h"

namespace sorei::nn::lr_sched {

class CosineAnnealingLR : public LRScheduler {
  public:
    CosineAnnealingLR(float start_lr, float end_lr, int max_steps)
        : LRScheduler(start_lr),
          start_lr_(start_lr),
          end_lr_(end_lr),
          max_steps_(max_steps - 1) {

        SOREI_CHECK(max_steps > 1);
    }

    float start_lr() const { return start_lr_; }
    float end_lr() const { return end_lr_; }
    int max_steps() const { return max_steps_ + 1; }

    std::string info() const override {
        return std::format(
            "CosineAnnealingLR(start_lr={:.6g}, end_lr={:.6g}, max_steps={})",
            start_lr_,
            end_lr_,
            max_steps_ + 1
        );
    }

  private:
    float start_lr_;
    float end_lr_;
    int max_steps_;

    float lr_update(int step) override {
        if (step >= max_steps_)
            return end_lr_;
        float t = static_cast<float>(step) / max_steps_;
        float lambda = 0.5f * (1.0f - std::cos(std::numbers::pi_v<float> * t));
        return start_lr_ + lambda * (end_lr_ - start_lr_);
    }
};

} // namespace sorei::nn::lr_sched
