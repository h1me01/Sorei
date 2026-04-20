#pragma once

#include <cmath>
#include <format>
#include <numbers>
#include <string>

#include "../misc.h"

namespace sorei::nn::lr_sched {

class LRScheduler {
  public:
    explicit LRScheduler(float lr)
        : lr_(lr) {}

    virtual ~LRScheduler() = default;

    void step() { lr_ = lr_update(++step_count_); }
    float get_lr() const { return lr_; }
    virtual std::string info() const = 0;

  protected:
    virtual float lr_update(int step) { return lr_; }

  private:
    float lr_;
    int step_count_ = 0;
};

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

struct ExponentialLR : StepLR {
    ExponentialLR(float lr, float gamma)
        : StepLR(lr, gamma, 1) {}

    std::string info() const override {
        return std::format("ExponentialLR(lr={:.6g}, gamma={:.6g})", base_lr(), gamma());
    }
};

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
