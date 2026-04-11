#pragma once

#include "adamw/adamw.h"

namespace nn::optim {

struct Adam : public AdamW {
    Adam(std::vector<layer::Param*> params, float beta1 = 0.9, float beta2 = 0.999)
        : AdamW(std::move(params), beta1, beta2, 0.0f) {}
};

} // namespace nn::optim
