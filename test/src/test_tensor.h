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

// CPUArray<float>

TEST(Tensor, CPUArray_DefaultConstruct) {
    CPUArray<float> a;
    EXPECT_EQ(a.size(), 0);
    EXPECT_TRUE(a.empty());
}

TEST(Tensor, CPUArray_SizedConstruct_ZeroInit) {
    CPUArray<float> a(8);
    EXPECT_EQ(a.size(), 8);
    EXPECT_FALSE(a.empty());
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(a[i], 0.0f);
}

TEST(Tensor, CPUArray_CopyDeep) {
    CPUArray<float> a(4);
    for (int i = 0; i < 4; ++i)
        a[i] = (float)i;

    CPUArray<float> b(a);
    EXPECT_EQ(b.size(), 4);

    a[0] = 99.0f;
    EXPECT_EQ(b[0], 0.0f);
}

TEST(Tensor, CPUArray_Fill) {
    CPUArray<float> a(5);
    a.fill(3.14f);
    for (int i = 0; i < 5; ++i)
        EXPECT_NEAR(a[i], 3.14f, 1e-6f);
}

TEST(Tensor, CPUArray_Clear) {
    CPUArray<float> a(4);
    a.fill(7.0f);
    a.clear();
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(a[i], 0.0f);
}

TEST(Tensor, CPUArray_Resize_Noop) {
    CPUArray<float> a(4);
    a.fill(1.0f);
    a.resize(4);
    EXPECT_EQ(a.size(), 4);
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(a[i], 1.0f);
}

TEST(Tensor, CPUArray_Resize_Realloc) {
    CPUArray<float> a(4);
    a.fill(1.0f);
    a.resize(8);
    EXPECT_EQ(a.size(), 8);
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(a[i], 0.0f);
}

TEST(Tensor, CPUArray_Bytes) {
    CPUArray<float> a(10);
    EXPECT_EQ(a.bytes(), 10 * sizeof(float));
}

// CPUMatrix<float>

TEST(Tensor, CPUMatrix_DefaultConstruct) {
    CPUMatrix<float> m;
    EXPECT_EQ(m.rows(), 0);
    EXPECT_EQ(m.cols(), 0);
    EXPECT_TRUE(m.empty());
}

TEST(Tensor, CPUMatrix_Construction) {
    CPUMatrix<float> m(Shape(3, 4));
    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 4);
    EXPECT_EQ(m.size(), 12);
    EXPECT_EQ(m.bytes(), 12 * sizeof(float));
}

TEST(Tensor, CPUMatrix_ColumnMajorLayout) {
    CPUMatrix<float> m(Shape(2, 3));
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

TEST(Tensor, CPUMatrix_Fill) {
    CPUMatrix<float> m(Shape(3, 3));
    m.fill(7.0f);
    for (int i = 0; i < 9; ++i)
        EXPECT_EQ(m(i), 7.0f);
}

TEST(Tensor, CPUMatrix_ArithmeticAdd) {
    CPUMatrix<float> a(Shape(2, 2)), b(Shape(2, 2));
    a.fill(3.0f);
    b.fill(2.0f);
    auto c = a + b;
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(c(i), 5.0f);
}

TEST(Tensor, CPUMatrix_ArithmeticSub) {
    CPUMatrix<float> a(Shape(2, 2)), b(Shape(2, 2));
    a.fill(5.0f);
    b.fill(3.0f);
    auto c = a - b;
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(c(i), 2.0f);
}

TEST(Tensor, CPUMatrix_ScalarMul) {
    CPUMatrix<float> m(Shape(2, 3));
    m.fill(2.0f);
    auto s = m * 3.0f;
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(s(i), 6.0f, 1e-6f);
}

TEST(Tensor, CPUMatrix_ScalarDiv) {
    CPUMatrix<float> m(Shape(2, 3));
    m.fill(6.0f);
    auto s = m / 2.0f;
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(s(i), 3.0f, 1e-6f);
}

TEST(Tensor, CPUMatrix_Reshape) {
    CPUMatrix<float> m(Shape(2, 3));
    for (int i = 0; i < 6; ++i)
        m(i) = (float)i;
    auto r = m.reshape(3, 2);
    EXPECT_EQ(r.rows(), 3);
    EXPECT_EQ(r.cols(), 2);

    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(r(i), (float)i);
}

TEST(Tensor, CPUMatrix_Transpose) {
    CPUMatrix<float> m(Shape(2, 3));
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

TEST(Tensor, CPUMatrix_Resize_Noop) {
    CPUMatrix<float> m(Shape(2, 3));
    m.fill(1.0f);
    m.resize(Shape(2, 3));
    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(m(i), 1.0f);
}

TEST(Tensor, CPUMatrix_Resize_Realloc) {
    CPUMatrix<float> m(Shape(2, 3));
    m.fill(1.0f);
    m.resize(Shape(4, 2));
    EXPECT_EQ(m.rows(), 4);
    EXPECT_EQ(m.cols(), 2);
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(m(i), 0.0f);
}

// GPUMatrix<float>

TEST(Tensor, GPUMatrix_Construction) {
    GPUMatrix<float> m(Shape(3, 4));
    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 4);
    EXPECT_EQ(m.size(), 12);
    EXPECT_FALSE(m.empty());
}

TEST(Tensor, GPUMatrix_ZeroInit) {
    GPUMatrix<float> m(Shape(4, 4));
    auto cpu = m.to_cpu();
    cudaDeviceSynchronize();
    for (int i = 0; i < 16; ++i)
        EXPECT_EQ(cpu(i), 0.0f);
}

TEST(Tensor, GPUMatrix_UploadDownload_RoundTrip) {
    CPUMatrix<float> src(Shape(3, 4));
    for (int i = 0; i < 12; ++i)
        src(i) = (float)i * 0.5f;

    GPUMatrix<float> gpu(Shape(3, 4));
    gpu.upload(src);

    CPUMatrix<float> dst(Shape(3, 4));
    gpu.download(dst);
    cudaDeviceSynchronize();

    for (int i = 0; i < 12; ++i)
        EXPECT_NEAR(dst(i), src(i), 1e-6f);
}

TEST(Tensor, GPUMatrix_ToCpu) {
    CPUMatrix<float> src(Shape(2, 5));
    src.fill(42.0f);
    auto gpu = GPUMatrix<float>::from_cpu(src);
    auto cpu = gpu.to_cpu();
    cudaDeviceSynchronize();
    for (int i = 0; i < 10; ++i)
        EXPECT_NEAR(cpu(i), 42.0f, 1e-6f);
}

TEST(Tensor, GPUMatrix_Copy) {
    CPUMatrix<float> src(Shape(3, 3));
    for (int i = 0; i < 9; ++i)
        src(i) = (float)i;
    auto a = GPUMatrix<float>::from_cpu(src);
    GPUMatrix<float> b(a);

    auto zeros = CPUMatrix<float>(Shape(3, 3));
    a.upload(zeros);

    auto b_cpu = b.to_cpu();
    cudaDeviceSynchronize();
    for (int i = 0; i < 9; ++i)
        EXPECT_NEAR(b_cpu(i), (float)i, 1e-6f);
}

TEST(Tensor, GPUMatrix_Clear) {
    CPUMatrix<float> src(Shape(2, 3));
    src.fill(5.0f);
    auto gpu = GPUMatrix<float>::from_cpu(src);
    gpu.clear();
    auto cpu = gpu.to_cpu();
    cudaDeviceSynchronize();
    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(cpu(i), 0.0f);
}

TEST(Tensor, GPUMatrix_Resize_Noop) {
    CPUMatrix<float> src(Shape(2, 3));
    src.fill(7.0f);
    auto gpu = GPUMatrix<float>::from_cpu(src);
    gpu.resize(Shape(2, 3));
    auto cpu = gpu.to_cpu();
    cudaDeviceSynchronize();
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(cpu(i), 7.0f, 1e-6f);
}

TEST(Tensor, GPUMatrix_Resize_Realloc) {
    CPUMatrix<float> src(Shape(2, 3));
    src.fill(7.0f);
    auto gpu = GPUMatrix<float>::from_cpu(src);
    gpu.resize(Shape(4, 2));
    EXPECT_EQ(gpu.rows(), 4);
    EXPECT_EQ(gpu.cols(), 2);
    auto cpu = gpu.to_cpu();
    cudaDeviceSynchronize();
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(cpu(i), 0.0f);
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
