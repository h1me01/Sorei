#pragma once

#include <cmath>
#include <format>
#include <string>

#include "lr_scheduler.h"
#include "step_lr.h"

namespace sorei::nn::lr_sched {

struct ExponentialLR : StepLR {
    ExponentialLR(float lr, float gamma)
        : StepLR(lr, gamma, 1) {}

    std::string info() const override {
        return std::format("ExponentialLR(lr={:.6g}, gamma={:.6g})", base_lr(), gamma());
    }
};

} // namespace sorei::nn::lr_sched
