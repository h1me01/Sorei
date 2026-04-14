#pragma once

#include <cuda_runtime.h>

#include "sorei/nn.h"

#include "framework.h"

using namespace sorei::tensor;

// Shape

TEST(Tensor, Shape_BasicProperties) {
    Shape s(3, 4);
    EXPECT_EQ(s.rows(), 3);
    EXPECT_EQ(s.cols(), 4);
    EXPECT_EQ(s.size(), 12);
}

TEST(Tensor, Shape_Equality) {
    EXPECT_TRUE(Shape(2, 5) == Shape(2, 5));
    EXPECT_FALSE(Shape(2, 5) == Shape(5, 2));
    EXPECT_TRUE(Shape(2, 5) != Shape(5, 2));
}

TEST(Tensor, Shape_ZeroSize) {
    Shape s(0, 0);
    EXPECT_EQ(s.size(), 0);
    Shape s2(3, 0);
    EXPECT_EQ(s2.size(), 0);
}

TEST(Tensor, Shape_Str) {
    std::string str = Shape(3, 4).str();
    EXPECT_FALSE(str.empty());
}

// HostArray<float>

TEST(Tensor, HostArray_DefaultConstruct) {
    HostArray<float> a;
    EXPECT_EQ(a.size(), 0);
    EXPECT_TRUE(a.empty());
}

TEST(Tensor, HostArray_SizedConstruct_ZeroInit) {
    HostArray<float> a(8);
    EXPECT_EQ(a.size(), 8);
    EXPECT_FALSE(a.empty());
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(a[i], 0.0f);
}

TEST(Tensor, HostArray_CopyDeep) {
    HostArray<float> a(4);
    for (int i = 0; i < 4; ++i)
        a[i] = (float)i;

    HostArray<float> b(a);
    EXPECT_EQ(b.size(), 4);

    a[0] = 99.0f;
    EXPECT_EQ(b[0], 0.0f);
}

TEST(Tensor, HostArray_Fill) {
    HostArray<float> a(5);
    a.fill(3.14f);
    for (int i = 0; i < 5; ++i)
        EXPECT_NEAR(a[i], 3.14f, 1e-6f);
}

TEST(Tensor, HostArray_Clear) {
    HostArray<float> a(4);
    a.fill(7.0f);
    a.clear();
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(a[i], 0.0f);
}

TEST(Tensor, HostArray_Resize_Noop) {
    HostArray<float> a(4);
    a.fill(1.0f);
    a.resize(4);
    EXPECT_EQ(a.size(), 4);
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(a[i], 1.0f);
}

TEST(Tensor, HostArray_Resize_Realloc) {
    HostArray<float> a(4);
    a.fill(1.0f);
    a.resize(8);
    EXPECT_EQ(a.size(), 8);
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(a[i], 0.0f);
}

TEST(Tensor, HostArray_Bytes) {
    HostArray<float> a(10);
    EXPECT_EQ(a.bytes(), 10 * sizeof(float));
}

// HostMatrix<float>

TEST(Tensor, HostMatrix_DefaultConstruct) {
    HostMatrix<float> m;
    EXPECT_EQ(m.rows(), 0);
    EXPECT_EQ(m.cols(), 0);
    EXPECT_TRUE(m.empty());
}

TEST(Tensor, HostMatrix_Construction) {
    HostMatrix<float> m(Shape(3, 4));
    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 4);
    EXPECT_EQ(m.size(), 12);
    EXPECT_EQ(m.bytes(), 12 * sizeof(float));
}

TEST(Tensor, HostMatrix_ColumnMajorLayout) {
    HostMatrix<float> m(Shape(2, 3));
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

TEST(Tensor, HostMatrix_Fill) {
    HostMatrix<float> m(Shape(3, 3));
    m.fill(7.0f);
    for (int i = 0; i < 9; ++i)
        EXPECT_EQ(m(i), 7.0f);
}

TEST(Tensor, HostMatrix_ArithmeticAdd) {
    HostMatrix<float> a(Shape(2, 2)), b(Shape(2, 2));
    a.fill(3.0f);
    b.fill(2.0f);
    auto c = a + b;
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(c(i), 5.0f);
}

TEST(Tensor, HostMatrix_ArithmeticSub) {
    HostMatrix<float> a(Shape(2, 2)), b(Shape(2, 2));
    a.fill(5.0f);
    b.fill(3.0f);
    auto c = a - b;
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(c(i), 2.0f);
}

TEST(Tensor, HostMatrix_ScalarMul) {
    HostMatrix<float> m(Shape(2, 3));
    m.fill(2.0f);
    auto s = m * 3.0f;
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(s(i), 6.0f, 1e-6f);
}

TEST(Tensor, HostMatrix_ScalarDiv) {
    HostMatrix<float> m(Shape(2, 3));
    m.fill(6.0f);
    auto s = m / 2.0f;
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(s(i), 3.0f, 1e-6f);
}

TEST(Tensor, HostMatrix_Reshape) {
    HostMatrix<float> m(Shape(2, 3));
    for (int i = 0; i < 6; ++i)
        m(i) = (float)i;
    auto r = m.reshape(3, 2);
    EXPECT_EQ(r.rows(), 3);
    EXPECT_EQ(r.cols(), 2);

    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(r(i), (float)i);
}

TEST(Tensor, HostMatrix_Transpose) {
    HostMatrix<float> m(Shape(2, 3));
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

TEST(Tensor, HostMatrix_Resize_Noop) {
    HostMatrix<float> m(Shape(2, 3));
    m.fill(1.0f);
    m.resize(Shape(2, 3));
    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(m(i), 1.0f);
}

TEST(Tensor, HostMatrix_Resize_Realloc) {
    HostMatrix<float> m(Shape(2, 3));
    m.fill(1.0f);
    m.resize(Shape(4, 2));
    EXPECT_EQ(m.rows(), 4);
    EXPECT_EQ(m.cols(), 2);
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(m(i), 0.0f);
}

// DeviceMatrix<float>

TEST(Tensor, DeviceMatrix_Construction) {
    DeviceMatrix<float> m(Shape(3, 4));
    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 4);
    EXPECT_EQ(m.size(), 12);
    EXPECT_FALSE(m.empty());
}

TEST(Tensor, DeviceMatrix_ZeroInit) {
    DeviceMatrix<float> m(Shape(4, 4));
    auto host = m.to_host();
    cudaDeviceSynchronize();
    for (int i = 0; i < 16; ++i)
        EXPECT_EQ(host(i), 0.0f);
}

TEST(Tensor, DeviceMatrix_UploadDownload_RoundTrip) {
    HostMatrix<float> src(Shape(3, 4));
    for (int i = 0; i < 12; ++i)
        src(i) = (float)i * 0.5f;

    DeviceMatrix<float> dev(Shape(3, 4));
    dev.upload(src);

    HostMatrix<float> dst(Shape(3, 4));
    dev.download(dst);
    cudaDeviceSynchronize();

    for (int i = 0; i < 12; ++i)
        EXPECT_NEAR(dst(i), src(i), 1e-6f);
}

TEST(Tensor, DeviceMatrix_ToCpu) {
    HostMatrix<float> src(Shape(2, 5));
    src.fill(42.0f);
    auto dev = DeviceMatrix<float>::from_host(src);
    auto host = dev.to_host();
    cudaDeviceSynchronize();
    for (int i = 0; i < 10; ++i)
        EXPECT_NEAR(host(i), 42.0f, 1e-6f);
}

TEST(Tensor, DeviceMatrix_Copy) {
    HostMatrix<float> src(Shape(3, 3));
    for (int i = 0; i < 9; ++i)
        src(i) = (float)i;
    auto a = DeviceMatrix<float>::from_host(src);
    DeviceMatrix<float> b(a);

    auto zeros = HostMatrix<float>(Shape(3, 3));
    a.upload(zeros);

    auto b_cpu = b.to_host();
    cudaDeviceSynchronize();
    for (int i = 0; i < 9; ++i)
        EXPECT_NEAR(b_cpu(i), (float)i, 1e-6f);
}

TEST(Tensor, DeviceMatrix_Clear) {
    HostMatrix<float> src(Shape(2, 3));
    src.fill(5.0f);
    auto dev = DeviceMatrix<float>::from_host(src);
    dev.clear();
    auto host = dev.to_host();
    cudaDeviceSynchronize();
    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(host(i), 0.0f);
}

TEST(Tensor, DeviceMatrix_Resize_Noop) {
    HostMatrix<float> src(Shape(2, 3));
    src.fill(7.0f);
    auto dev = DeviceMatrix<float>::from_host(src);
    dev.resize(Shape(2, 3));
    auto host = dev.to_host();
    cudaDeviceSynchronize();
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(host(i), 7.0f, 1e-6f);
}

TEST(Tensor, DeviceMatrix_Resize_Realloc) {
    HostMatrix<float> src(Shape(2, 3));
    src.fill(7.0f);
    auto dev = DeviceMatrix<float>::from_host(src);
    dev.resize(Shape(4, 2));
    EXPECT_EQ(dev.rows(), 4);
    EXPECT_EQ(dev.cols(), 2);
    auto host = dev.to_host();
    cudaDeviceSynchronize();
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(host(i), 0.0f);
}

// nn::Tensor<T>

TEST(Tensor, NNTensor_1D_Construction) {
    sorei::nn::Tensor<float> t({8});
    EXPECT_EQ(t.size(), 8);
}

TEST(Tensor, NNTensor_1D_Indexing) {
    sorei::nn::Tensor<float> t({4});
    t[0] = 1.0f;
    t[1] = 2.0f;
    t[2] = 3.0f;
    t[3] = 4.0f;
    EXPECT_NEAR(t[0], 1.0f, 1e-6f);
    EXPECT_NEAR(t[3], 4.0f, 1e-6f);
}

TEST(Tensor, NNTensor_1D_Fill) {
    sorei::nn::Tensor<float> t({6});
    t.fill(3.3f);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(t[i], 3.3f, 1e-5f);
}

TEST(Tensor, NNTensor_2D_Construction) {
    sorei::nn::Tensor<float> t({4, 3});
    EXPECT_EQ(t.size(), 12);
}

TEST(Tensor, NNTensor_2D_Indexing_ColumnMajor) {
    sorei::nn::Tensor<float> t({3, 2});
    t(0, 0) = 10.0f;
    t(1, 0) = 20.0f;
    t(2, 0) = 30.0f;
    t(0, 1) = 40.0f;
    t(1, 1) = 50.0f;
    t(2, 1) = 60.0f;

    EXPECT_NEAR(t(0, 0), 10.0f, 1e-6f);
    EXPECT_NEAR(t(2, 1), 60.0f, 1e-6f);
}

TEST(Tensor, NNTensor_Resize) {
    sorei::nn::Tensor<float> t({4});
    t.resize({8});
    EXPECT_EQ(t.size(), 8);
}

TEST(Tensor, NNTensor_Int) {
    sorei::nn::Tensor<int> t({5});
    t[0] = 3;
    t[4] = 7;
    EXPECT_EQ(t[0], 3);
    EXPECT_EQ(t[4], 7);
}
