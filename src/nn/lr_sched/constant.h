#pragma once

#include <format>
#include <string>

#include "lr_scheduler.h"

namespace sorei::nn::lr_sched {

struct Constant : public LRScheduler {
    explicit Constant(float lr)
        : LRScheduler(lr) {}

    std::string info() const override { return std::format("Constant(lr={:.6g})", get()); }
};

} // namespace sorei::nn::lr_sched
