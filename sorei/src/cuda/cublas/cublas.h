#pragma once

#include <cublas_v2.h>
#include <cusparse_v2.h>

#include "../util.h"

namespace sorei::cuda::cublas {

void sgemm(
    bool trans_a,
    bool trans_b,
    float alpha,
    const tensor::DeviceMatrix<float>& a,
    const tensor::DeviceMatrix<float>& b,
    float beta,
    tensor::DeviceMatrix<float>& c
);

} // namespace sorei::cuda::cublas
