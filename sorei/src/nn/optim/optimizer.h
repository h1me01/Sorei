#pragma once

#include <vector>

#include "../../util.h"
#include "../layer/param.h"

namespace sorei::nn {

class Optimizer {
  public:
    Optimizer(std::vector<Param*> params)
        : params_(std::move(params)) {}

    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = delete;
    Optimizer(Optimizer&&) = delete;
    Optimizer& operator=(Optimizer&&) = delete;

    virtual ~Optimizer() = default;

    virtual void step(float lr) = 0;

  protected:
    std::vector<Param*> params_;
};

} // namespace sorei::nn
