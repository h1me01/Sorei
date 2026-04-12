#pragma once

#include <cmath>
#include <format>
#include <numbers>
#include <string>

#include "lr_scheduler.h"

namespace sorei::nn::lr_sched {

class CosineAnnealing : public LRScheduler {
  public:
    CosineAnnealing(float start, float end, int max_steps)
        : LRScheduler(start),
          start_(start),
          end_(end),
          max_steps_(max_steps - 1) {

        SOREI_CHECK(max_steps > 1);
    }

    std::string info() const override {
        return std::format(
            "CosineAnnealing(start={:.6g}, end={:.6g}, max_steps={})", start_, end_, max_steps_ + 1
        );
    }

  private:
    float start_, end_;
    int max_steps_;

    float lr_update(int step) override {
        if (step >= max_steps_)
            return end_;
        float t = static_cast<float>(step) / max_steps_;
        float lambda = 0.5f * (1.0f - std::cos(std::numbers::pi_v<float> * t));
        return start_ + lambda * (end_ - start_);
    }
};

} // namespace sorei::nn::lr_sched
