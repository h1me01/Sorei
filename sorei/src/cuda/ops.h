#pragma once

#include "../matrix/include.h"
#include "utils.h"

namespace sorei::cuda {

void add(
    const matrix::DeviceMatrix<float>& a,
    const matrix::DeviceMatrix<float>& b,
    matrix::DeviceMatrix<float>& out
);

void set(matrix::DeviceMatrix<float>& a, const float v);

} // namespace sorei::cuda
