#pragma once

#include <string>
#include <vector>

#include "../../cuda/include.h"
#include "../layer/param.h"

namespace sorei::nn::optim {

class Optimizer {
  public:
    Optimizer(std::vector<layer::Param*> params)
        : params_(std::move(params)) {}

    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = delete;
    Optimizer(Optimizer&&) = delete;
    Optimizer& operator=(Optimizer&&) = delete;

    virtual ~Optimizer() = default;

    virtual void load_state(const std::string& path) {}
    virtual void save_state(const std::string& path) const {}

    virtual void step(float lr) = 0;

  protected:
    std::vector<layer::Param*> params_;
};

} // namespace sorei::nn::optim
