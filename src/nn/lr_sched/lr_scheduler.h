#pragma once

#include <string>

#include "../../misc.h"

namespace nn::lr_sched {

class LRScheduler {
  public:
    explicit LRScheduler(float lr)
        : lr_(lr) {}

    virtual ~LRScheduler() = default;

    void step() { lr_ = lr_update(++step_count_); }
    float get() const { return lr_; }
    virtual std::string info() const = 0;

  protected:
    virtual float lr_update(int step) { return lr_; }

  private:
    float lr_;
    int step_count_ = 0;
};

} // namespace nn::lr_sched
