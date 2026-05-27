#pragma once

#include <cmath>
#include <cuda_runtime.h>

#include "sorei/nn.h"

#include "framework.h"

using namespace sorei::matrix;
using namespace sorei::cuda;
using namespace sorei::nn;

namespace {

static DeviceMatrix<float>
make_device(const Shape& shape, std::initializer_list<float> row_major_vals) {
    HostMatrix<float> m(shape);
    int idx = 0;
    for (float v : row_major_vals) {
        int r = idx / shape.cols();
        int c = idx % shape.cols();
        m(r, c) = v;
        ++idx;
    }
    return DeviceMatrix<float>::from_host(m);
}

} // namespace

TEST(Kernels, SGemm_Basic) {
    auto A = make_device({2, 3}, {1, 2, 3, 4, 5, 6});
    auto B = make_device({3, 2}, {7, 8, 9, 10, 11, 12});
    DeviceMatrix<float> C({2, 2});

    cublas::sgemm(false, false, 1.0f, A, B, 0.0f, C);
    cudaDeviceSynchronize();
    auto host = C.to_host();

    EXPECT_NEAR(host(0, 0), 58.0f, 1e-3f);
    EXPECT_NEAR(host(1, 0), 139.0f, 1e-3f);
    EXPECT_NEAR(host(0, 1), 64.0f, 1e-3f);
    EXPECT_NEAR(host(1, 1), 154.0f, 1e-3f);
}

TEST(Kernels, SGemm_TransposeA) {
    auto A = make_device({3, 2}, {1, 2, 3, 4, 5, 6});
    auto B = make_device({3, 2}, {7, 8, 9, 10, 11, 12});
    DeviceMatrix<float> C({2, 2});

    cublas::sgemm(true, false, 1.0f, A, B, 0.0f, C);
    cudaDeviceSynchronize();
    auto host = C.to_host();

    EXPECT_NEAR(host(0, 0), 89.0f, 1e-3f);
    EXPECT_NEAR(host(1, 0), 116.0f, 1e-3f);
    EXPECT_NEAR(host(0, 1), 98.0f, 1e-3f);
    EXPECT_NEAR(host(1, 1), 128.0f, 1e-3f);
}

TEST(Kernels, SGemm_TransposeB) {
    auto A = make_device({2, 3}, {1, 2, 3, 4, 5, 6});
    auto B = make_device({2, 3}, {7, 8, 9, 10, 11, 12});
    DeviceMatrix<float> C({2, 2});

    cublas::sgemm(false, true, 1.0f, A, B, 0.0f, C);
    cudaDeviceSynchronize();
    auto host = C.to_host();

    EXPECT_NEAR(host(0, 0), 50.0f, 1e-3f);
    EXPECT_NEAR(host(1, 0), 122.0f, 1e-3f);
    EXPECT_NEAR(host(0, 1), 68.0f, 1e-3f);
    EXPECT_NEAR(host(1, 1), 167.0f, 1e-3f);
}

TEST(Kernels, Add_Basic) {
    auto a = make_device({2, 3}, {1, 2, 3, 4, 5, 6});
    auto b = make_device({2, 3}, {10, 20, 30, 40, 50, 60});
    DeviceMatrix<float> out({2, 3});
    sorei::cuda::add(a, b, out);
    cudaDeviceSynchronize();
    auto h = out.to_host();
    EXPECT_NEAR(h(0, 0), 11.0f, 1e-3f);
    EXPECT_NEAR(h(0, 1), 22.0f, 1e-3f);
    EXPECT_NEAR(h(0, 2), 33.0f, 1e-3f);
    EXPECT_NEAR(h(1, 0), 44.0f, 1e-3f);
    EXPECT_NEAR(h(1, 1), 55.0f, 1e-3f);
    EXPECT_NEAR(h(1, 2), 66.0f, 1e-3f);
}

TEST(Kernels, Set_Basic) {
    DeviceMatrix<float> a({3, 4});
    sorei::cuda::set(a, 7.0f);
    cudaDeviceSynchronize();
    auto h = a.to_host();
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            EXPECT_NEAR(h(r, c), 7.0f, 1e-3f);
}
