#pragma once

#include "../cublas/cublas.h"
#include "../util.h"

namespace kernel {

void mat_mul_forward(
    const data::GPUMatrix<float>& weight,
    const data::GPUMatrix<float>& in,
    data::GPUMatrix<float>& out
);

void mat_mul_backward(
    const data::GPUMatrix<float>& weight,
    data::GPUMatrix<float>& weight_g,
    const data::GPUMatrix<float>& in,
    data::GPUMatrix<float>& in_g,
    const data::GPUMatrix<float>& out_g
);

} // namespace kernel
