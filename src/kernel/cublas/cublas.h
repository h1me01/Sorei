#pragma once

#include <cublas_v2.h>
#include <cusparse_v2.h>

#include "../util.h"

namespace kernel::cublas {

void create();
void destroy();

void sgemm(
    bool trans_a,
    bool trans_b,
    float alpha,
    const tensor::GPUMatrix<float>& a,
    const tensor::GPUMatrix<float>& b,
    float beta,
    tensor::GPUMatrix<float>& c
);

} // namespace kernel::cublas
