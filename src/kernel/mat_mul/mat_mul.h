#pragma once

#include "../cublas/cublas.h"
#include "../util.h"

namespace kernel {

void mat_mul_forward(
    const tensor::GPUMatrix<float>& weight,
    const tensor::GPUMatrix<float>& in,
    tensor::GPUMatrix<float>& out
);

void mat_mul_backward(
    const tensor::GPUMatrix<float>& weight,
    tensor::GPUMatrix<float>& weight_g,
    const tensor::GPUMatrix<float>& in,
    tensor::GPUMatrix<float>& in_g,
    const tensor::GPUMatrix<float>& out_g
);

} // namespace kernel
