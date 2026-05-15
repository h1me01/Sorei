#pragma once

#include <cuda_runtime.h>

#include "framework.h"
#include "sorei/nn.h"

using namespace sorei::matrix;

// Shape

TEST(Matrix, Shape_BasicProperties) {
    Shape s(3, 4);
    EXPECT_EQ(s.rows(), 3);
    EXPECT_EQ(s.cols(), 4);
    EXPECT_EQ(s.size(), 12);
}

TEST(Matrix, Shape_Equality) {
    EXPECT_TRUE(Shape(2, 5) == Shape(2, 5));
    EXPECT_FALSE(Shape(2, 5) == Shape(5, 2));
    EXPECT_TRUE(Shape(2, 5) != Shape(5, 2));
}

TEST(Matrix, Shape_ZeroSize) {
    Shape s(0, 0);
    EXPECT_EQ(s.size(), 0);
    Shape s2(3, 0);
    EXPECT_EQ(s2.size(), 0);
}

TEST(Matrix, Shape_Str) {
    std::string str = Shape(3, 4).str();
    EXPECT_FALSE(str.empty());
}

// HostMatrix<float>

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

TEST(Matrix, HostMatrix_ArithmeticAdd) {
    HostMatrix<float> a({2, 2}), b({2, 2});
    a.fill(3.0f);
    b.fill(2.0f);
    auto c = a + b;
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(c(i), 5.0f);
}

TEST(Matrix, HostMatrix_ArithmeticSub) {
    HostMatrix<float> a({2, 2}), b({2, 2});
    a.fill(5.0f);
    b.fill(3.0f);
    auto c = a - b;
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(c(i), 2.0f);
}

TEST(Matrix, HostMatrix_ScalarMul) {
    HostMatrix<float> m({2, 3});
    m.fill(2.0f);
    auto s = m * 3.0f;
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(s(i), 6.0f, 1e-6f);
}

TEST(Matrix, HostMatrix_ScalarDiv) {
    HostMatrix<float> m({2, 3});
    m.fill(6.0f);
    auto s = m / 2.0f;
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(s(i), 3.0f, 1e-6f);
}

TEST(Matrix, HostMatrix_Reshape) {
    HostMatrix<float> m({2, 3});
    for (int i = 0; i < 6; ++i)
        m(i) = (float)i;
    auto r = m.reshape({3, 2});
    EXPECT_EQ(r.rows(), 3);
    EXPECT_EQ(r.cols(), 2);

    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(r(i), (float)i);
}

TEST(Matrix, HostMatrix_Transpose) {
    HostMatrix<float> m({2, 3});
    m(0, 0) = 1;
    m(0, 1) = 2;
    m(0, 2) = 3;
    m(1, 0) = 4;
    m(1, 1) = 5;
    m(1, 2) = 6;

    auto t = m.transpose();
    EXPECT_EQ(t.rows(), 3);
    EXPECT_EQ(t.cols(), 2);
    EXPECT_EQ(t(0, 0), 1.0f);
    EXPECT_EQ(t(1, 0), 2.0f);
    EXPECT_EQ(t(2, 0), 3.0f);
    EXPECT_EQ(t(0, 1), 4.0f);
    EXPECT_EQ(t(1, 1), 5.0f);
    EXPECT_EQ(t(2, 1), 6.0f);
}

TEST(Matrix, HostMatrix_Resize_Noop) {
    HostMatrix<float> m({2, 3});
    m.fill(1.0f);
    m.resize({2, 3});
    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(m(i), 1.0f);
}

TEST(Matrix, HostMatrix_Resize_Realloc) {
    HostMatrix<float> m({2, 3});
    m.fill(1.0f);
    m.resize({4, 2});
    EXPECT_EQ(m.rows(), 4);
    EXPECT_EQ(m.cols(), 2);
}

// DeviceMatrix<float>

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

TEST(Matrix, DeviceMatrix_Resize_Noop) {
    HostMatrix<float> src({2, 3});
    src.fill(7.0f);
    auto dev = DeviceMatrix<float>::from_host(src);
    dev.resize({2, 3});
    auto host = dev.to_host();
    cudaDeviceSynchronize();
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(host(i), 7.0f, 1e-6f);
}

TEST(Matrix, DeviceMatrix_Resize_Realloc) {
    HostMatrix<float> src({2, 3});
    src.fill(7.0f);
    auto dev = DeviceMatrix<float>::from_host(src);
    dev.resize({4, 2});
    EXPECT_EQ(dev.rows(), 4);
    EXPECT_EQ(dev.cols(), 2);
}
