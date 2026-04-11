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
    const data::GPUMatrix<float>& a,
    const data::GPUMatrix<float>& b,
    float beta,
    data::GPUMatrix<float>& c
);

} // namespace kernel::cublas
