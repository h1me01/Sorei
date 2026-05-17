#pragma once

#include <cuda_runtime.h>

#include "framework.h"
#include "sorei/nn.h"

using namespace sorei::matrix;

TEST(Matrix, HostMatrix_DefaultConstruct) {
    HostMatrix<float> m;
    EXPECT_EQ(m.rows(), 0);
    EXPECT_EQ(m.cols(), 0);
    EXPECT_TRUE(m.empty());
}

TEST(Matrix, HostMatrix_Construction) {
    HostMatrix<float> m({3, 4});
    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 4);
    EXPECT_EQ(m.size(), 12);
    EXPECT_EQ(m.bytes(), 12 * sizeof(float));
}

TEST(Matrix, HostMatrix_ColumnMajorLayout) {
    HostMatrix<float> m({2, 3});
    m(0, 0) = 1.0f;
    m(1, 0) = 2.0f;
    m(0, 1) = 3.0f;
    m(1, 1) = 4.0f;
    m(0, 2) = 5.0f;
    m(1, 2) = 6.0f;

    EXPECT_EQ(m(0), 1.0f);
    EXPECT_EQ(m(1), 2.0f);
    EXPECT_EQ(m(2), 3.0f);
    EXPECT_EQ(m(3), 4.0f);
    EXPECT_EQ(m(4), 5.0f);
    EXPECT_EQ(m(5), 6.0f);
}

TEST(Matrix, HostMatrix_Fill) {
    HostMatrix<float> m({3, 3});
    m.fill(7.0f);
    for (int i = 0; i < 9; ++i)
        EXPECT_EQ(m(i), 7.0f);
}

TEST(Matrix, HostMatrix_Resize_Realloc) {
    HostMatrix<float> m({2, 3});
    m.fill(1.0f);
    m.resize({4, 2});
    EXPECT_EQ(m.rows(), 4);
    EXPECT_EQ(m.cols(), 2);
}

TEST(Matrix, DeviceMatrix_Construction) {
    DeviceMatrix<float> m({3, 4});
    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 4);
    EXPECT_EQ(m.size(), 12);
    EXPECT_FALSE(m.empty());
}

TEST(Matrix, DeviceMatrix_UploadDownload_RoundTrip) {
    HostMatrix<float> src({3, 4});
    for (int i = 0; i < 12; ++i)
        src(i) = (float)i * 0.5f;

    DeviceMatrix<float> dev({3, 4});
    dev.upload(src);

    HostMatrix<float> dst({3, 4});
    dev.download(dst);
    cudaDeviceSynchronize();

    for (int i = 0; i < 12; ++i)
        EXPECT_NEAR(dst(i), src(i), 1e-6f);
}

TEST(Matrix, DeviceMatrix_ToCpu) {
    HostMatrix<float> src({2, 5});
    src.fill(42.0f);
    auto dev = DeviceMatrix<float>::from_host(src);
    auto host = dev.to_host();
    cudaDeviceSynchronize();
    for (int i = 0; i < 10; ++i)
        EXPECT_NEAR(host(i), 42.0f, 1e-6f);
}

TEST(Matrix, DeviceMatrix_Copy) {
    HostMatrix<float> src({3, 3});
    for (int i = 0; i < 9; ++i)
        src(i) = (float)i;
    auto a = DeviceMatrix<float>::from_host(src);
    DeviceMatrix<float> b(a);

    auto zeros = HostMatrix<float>({3, 3});
    a.upload(zeros);

    auto b_cpu = b.to_host();
    cudaDeviceSynchronize();
    for (int i = 0; i < 9; ++i)
        EXPECT_NEAR(b_cpu(i), (float)i, 1e-6f);
}

TEST(Matrix, DeviceMatrix_Clear) {
    HostMatrix<float> src({2, 3});
    src.fill(5.0f);
    auto dev = DeviceMatrix<float>::from_host(src);
    dev.clear();
    auto host = dev.to_host();
    cudaDeviceSynchronize();
    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(host(i), 0.0f);
}

TEST(Matrix, DeviceMatrix_Resize_Realloc) {
    HostMatrix<float> src({2, 3});
    src.fill(7.0f);
    auto dev = DeviceMatrix<float>::from_host(src);
    dev.resize({4, 2});
    EXPECT_EQ(dev.rows(), 4);
    EXPECT_EQ(dev.cols(), 2);
}
