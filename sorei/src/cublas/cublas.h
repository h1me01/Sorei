#pragma once

#include "../matrix/include.h"
#include "../util.h"

namespace sorei::cublas {

void sgemm(
    bool trans_a,
    bool trans_b,
    float alpha,
    const matrix::DeviceMatrix<float>& a,
    const matrix::DeviceMatrix<float>& b,
    float beta,
    matrix::DeviceMatrix<float>& c
);

} // namespace sorei::cublas
