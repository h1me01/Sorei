#pragma once

#include <cmath>
#include <cuda_runtime.h>

#include "sorei/nn.h"

#include "framework.h"

using namespace sorei::tensor;
using namespace sorei::cuda;
using namespace sorei::nn::layer;

// set

TEST(Kernels, Set_FillsAllElements) {
    GPUMatrix<float> m(Shape(4, 3));
    set(m, 7.5f);
    cudaDeviceSynchronize();
    auto cpu = m.to_cpu();
    for (int i = 0; i < 12; ++i)
        EXPECT_NEAR(cpu(i), 7.5f, 1e-6f);
}

TEST(Kernels, Set_ZeroFill) {
    GPUMatrix<float> m(Shape(5, 5));
    set(m, 1.0f);
    set(m, 0.0f);
    cudaDeviceSynchronize();
    auto cpu = m.to_cpu();
    for (int i = 0; i < 25; ++i)
        EXPECT_EQ(cpu(i), 0.0f);
}

// cublas::sgemm

// build a GPUMatrix from a flat row-major initializer
static GPUMatrix<float> make_gpu(int rows, int cols, std::initializer_list<float> row_major_vals) {
    CPUMatrix<float> m(Shape(rows, cols));
    int idx = 0;
    for (float v : row_major_vals) {
        int r = idx / cols;
        int c = idx % cols;
        m(r, c) = v;
        ++idx;
    }
    return GPUMatrix<float>::from_cpu(m);
}

TEST(Kernels, SGemm_Basic) {
    auto A = make_gpu(2, 3, {1, 2, 3, 4, 5, 6});
    auto B = make_gpu(3, 2, {7, 8, 9, 10, 11, 12});
    GPUMatrix<float> C(Shape(2, 2));

    cublas::sgemm(false, false, 1.0f, A, B, 0.0f, C);
    cudaDeviceSynchronize();
    auto cpu = C.to_cpu();

    EXPECT_NEAR(cpu(0, 0), 58.0f, 1e-3f);
    EXPECT_NEAR(cpu(1, 0), 139.0f, 1e-3f);
    EXPECT_NEAR(cpu(0, 1), 64.0f, 1e-3f);
    EXPECT_NEAR(cpu(1, 1), 154.0f, 1e-3f);
}

TEST(Kernels, SGemm_AlphaBeta) {
    auto A = make_gpu(2, 2, {1, 0, 0, 1});
    auto B = make_gpu(2, 2, {3, 4, 5, 6});
    GPUMatrix<float> C(Shape(2, 2));

    cublas::sgemm(false, false, 1.0f, A, B, 0.0f, C);
    cudaDeviceSynchronize();

    cublas::sgemm(false, false, 2.0f, A, B, 0.5f, C);
    cudaDeviceSynchronize();
    auto cpu = C.to_cpu();

    EXPECT_NEAR(cpu(0, 0), 7.5f, 1e-3f);
    EXPECT_NEAR(cpu(1, 0), 12.5f, 1e-3f);
    EXPECT_NEAR(cpu(0, 1), 10.0f, 1e-3f);
    EXPECT_NEAR(cpu(1, 1), 15.0f, 1e-3f);
}

TEST(Kernels, SGemm_TransposeA) {
    auto A = make_gpu(3, 2, {1, 2, 3, 4, 5, 6});
    auto B = make_gpu(3, 2, {7, 8, 9, 10, 11, 12});
    GPUMatrix<float> C(Shape(2, 2));

    cublas::sgemm(true, false, 1.0f, A, B, 0.0f, C);
    cudaDeviceSynchronize();
    auto cpu = C.to_cpu();

    EXPECT_NEAR(cpu(0, 0), 89.0f, 1e-3f);
    EXPECT_NEAR(cpu(1, 0), 116.0f, 1e-3f);
    EXPECT_NEAR(cpu(0, 1), 98.0f, 1e-3f);
    EXPECT_NEAR(cpu(1, 1), 128.0f, 1e-3f);
}

// ElemwiseUnary

static GPUMatrix<float>
run_unary_fwd(std::initializer_list<float> vals, const ElemwiseUnary::Op& op) {
    int n = vals.size();
    CPUMatrix<float> cpu_in(Shape(n, 1));
    int i = 0;
    for (float v : vals)
        cpu_in(i++, 0) = v;

    auto gpu_in = GPUMatrix<float>::from_cpu(cpu_in);
    GPUMatrix<float> gpu_out(Shape(n, 1));
    ElemwiseUnary::forward(gpu_in, gpu_out, op);
    cudaDeviceSynchronize();

    return gpu_out;
}

TEST(Kernels, UnaryFwd_Identity) {
    auto out = run_unary_fwd({-1.0f, 0.0f, 2.5f}, Identity{});
    auto cpu = out.to_cpu();
    EXPECT_NEAR(cpu(0), -1.0f, 1e-6f);
    EXPECT_NEAR(cpu(1), 0.0f, 1e-6f);
    EXPECT_NEAR(cpu(2), 2.5f, 1e-6f);
}

TEST(Kernels, UnaryFwd_ReLU) {
    auto out = run_unary_fwd({-2.0f, 0.0f, 1.5f}, ReLU{});
    auto cpu = out.to_cpu();
    EXPECT_NEAR(cpu(0), 0.0f, 1e-6f);
    EXPECT_NEAR(cpu(1), 0.0f, 1e-6f);
    EXPECT_NEAR(cpu(2), 1.5f, 1e-6f);
}

TEST(Kernels, UnaryFwd_ClampedReLU) {
    auto out = run_unary_fwd({-1.0f, 0.5f, 2.0f}, ClampedReLU{});
    auto cpu = out.to_cpu();
    EXPECT_NEAR(cpu(0), 0.0f, 1e-6f);
    EXPECT_NEAR(cpu(1), 0.5f, 1e-6f);
    EXPECT_NEAR(cpu(2), 1.0f, 1e-6f);
}

TEST(Kernels, UnaryFwd_SquaredClampedReLU) {
    auto out = run_unary_fwd({-1.0f, 0.5f, 2.0f}, SquaredClampedReLU{});
    auto cpu = out.to_cpu();
    EXPECT_NEAR(cpu(0), 0.0f, 1e-6f);
    EXPECT_NEAR(cpu(1), 0.25f, 1e-5f);
    EXPECT_NEAR(cpu(2), 1.0f, 1e-6f);
}

TEST(Kernels, UnaryFwd_Sigmoid) {
    auto out = run_unary_fwd({0.0f}, Sigmoid{});
    auto cpu = out.to_cpu();
    EXPECT_NEAR(cpu(0), 0.5f, 1e-5f);

    auto out2 = run_unary_fwd({10.0f, -10.0f}, Sigmoid{});
    auto cpu2 = out2.to_cpu();
    EXPECT_GT((double)cpu2(0), 0.99);
    EXPECT_LT((double)cpu2(1), 0.01);
}

TEST(Kernels, UnaryFwd_Abs) {
    auto out = run_unary_fwd({-3.0f, 0.0f, 2.0f}, Abs{});
    auto cpu = out.to_cpu();
    EXPECT_NEAR(cpu(0), 3.0f, 1e-6f);
    EXPECT_NEAR(cpu(1), 0.0f, 1e-6f);
    EXPECT_NEAR(cpu(2), 2.0f, 1e-6f);
}

TEST(Kernels, UnaryFwd_PowInt) {
    auto out = run_unary_fwd({2.0f, -3.0f}, PowInt{3});
    auto cpu = out.to_cpu();
    EXPECT_NEAR(cpu(0), 8.0f, 1e-4f);
    EXPECT_NEAR(cpu(1), -27.0f, 1e-4f);
}

TEST(Kernels, UnaryFwd_PowFloat) {
    auto out = run_unary_fwd({4.0f, 9.0f}, PowFloat{0.5f});
    auto cpu = out.to_cpu();
    EXPECT_NEAR(cpu(0), 2.0f, 1e-4f);
    EXPECT_NEAR(cpu(1), 3.0f, 1e-4f);
}

TEST(Kernels, UnaryFwd_Clamp) {
    auto out = run_unary_fwd({-5.0f, 0.3f, 5.0f}, Clamp{-1.0f, 1.0f});
    auto cpu = out.to_cpu();
    EXPECT_NEAR(cpu(0), -1.0f, 1e-6f);
    EXPECT_NEAR(cpu(1), 0.3f, 1e-6f);
    EXPECT_NEAR(cpu(2), 1.0f, 1e-6f);
}

TEST(Kernels, UnaryFwd_AddScaleUnary) {
    auto out = run_unary_fwd({1.0f, -2.0f}, AddScaleUnary{2.0f, 3.0f});
    auto cpu = out.to_cpu();
    EXPECT_NEAR(cpu(0), 5.0f, 1e-5f);
    EXPECT_NEAR(cpu(1), -1.0f, 1e-5f);
}

TEST(Kernels, UnaryFwd_DivLeftUnary) {
    auto out = run_unary_fwd({2.0f, 4.0f}, DivLeftUnary{8.0f});
    auto cpu = out.to_cpu();
    EXPECT_NEAR(cpu(0), 4.0f, 1e-5f);
    EXPECT_NEAR(cpu(1), 2.0f, 1e-5f);
}

// ElemwiseUnary backward

static CPUMatrix<float> run_unary_bwd(
    std::initializer_list<float> inputs,
    std::initializer_list<float> out_grads,
    const ElemwiseUnary::Op& op
) {
    int n = inputs.size();
    CPUMatrix<float> cpu_in(Shape(n, 1));
    CPUMatrix<float> cpu_og(Shape(n, 1));
    int i = 0;
    for (float v : inputs)
        cpu_in(i++, 0) = v;
    i = 0;
    for (float v : out_grads)
        cpu_og(i++, 0) = v;

    auto gpu_in = GPUMatrix<float>::from_cpu(cpu_in);
    GPUMatrix<float> gpu_in_g(Shape(n, 1));
    gpu_in_g.clear();
    auto gpu_og = GPUMatrix<float>::from_cpu(cpu_og);

    ElemwiseUnary::backward(gpu_in, gpu_in_g, gpu_og, op);
    cudaDeviceSynchronize();
    return gpu_in_g.to_cpu();
}

TEST(Kernels, UnaryBwd_ReLU_Positive) {
    auto g = run_unary_bwd({2.0f}, {3.0f}, ReLU{});
    EXPECT_NEAR(g(0), 3.0f, 1e-5f);
}

TEST(Kernels, UnaryBwd_ReLU_Negative) {
    auto g = run_unary_bwd({-1.0f}, {3.0f}, ReLU{});
    EXPECT_NEAR(g(0), 0.0f, 1e-5f);
}

TEST(Kernels, UnaryBwd_Sigmoid) {
    auto g = run_unary_bwd({0.0f}, {2.0f}, Sigmoid{});
    EXPECT_NEAR(g(0), 0.5f, 1e-4f);
}

TEST(Kernels, UnaryBwd_Abs) {
    auto g_pos = run_unary_bwd({3.0f}, {1.0f}, Abs{});
    auto g_neg = run_unary_bwd({-2.0f}, {1.0f}, Abs{});
    EXPECT_NEAR(g_pos(0), 1.0f, 1e-5f);
    EXPECT_NEAR(g_neg(0), -1.0f, 1e-5f);
}

TEST(Kernels, UnaryBwd_PowInt2) {
    auto g = run_unary_bwd({3.0f}, {1.0f}, PowInt{2});
    EXPECT_NEAR(g(0), 6.0f, 1e-4f);
}

TEST(Kernels, UnaryBwd_AddScale) {
    auto g = run_unary_bwd({5.0f}, {1.0f}, AddScaleUnary{4.0f, 0.0f});
    EXPECT_NEAR(g(0), 4.0f, 1e-5f);
}

// ElemwiseBinary forward

static CPUMatrix<float> run_binary_fwd(
    std::initializer_list<float> a_vals,
    std::initializer_list<float> b_vals,
    int rows,
    int cols,
    const ElemwiseBinary::Op& op
) {
    CPUMatrix<float> ca(Shape(rows, cols)), cb(Shape(rows, cols));
    int i = 0;
    for (float v : a_vals)
        ca(i++, 0) = v;
    i = 0;
    for (float v : b_vals)
        cb(i++, 0) = v;

    auto ga = GPUMatrix<float>::from_cpu(ca);
    auto gb = GPUMatrix<float>::from_cpu(cb);
    GPUMatrix<float> gc(Shape(rows, cols));

    ElemwiseBinary::forward(ga, gb, gc, op);
    cudaDeviceSynchronize();
    return gc.to_cpu();
}

TEST(Kernels, BinaryFwd_Add) {
    auto c = run_binary_fwd({1, 2, 3}, {4, 5, 6}, 3, 1, AddBinary{});
    EXPECT_NEAR(c(0), 5.0f, 1e-5f);
    EXPECT_NEAR(c(1), 7.0f, 1e-5f);
    EXPECT_NEAR(c(2), 9.0f, 1e-5f);
}

TEST(Kernels, BinaryFwd_Sub) {
    auto c = run_binary_fwd({5, 5, 5}, {1, 2, 3}, 3, 1, SubBinary{});
    EXPECT_NEAR(c(0), 4.0f, 1e-5f);
    EXPECT_NEAR(c(1), 3.0f, 1e-5f);
    EXPECT_NEAR(c(2), 2.0f, 1e-5f);
}

TEST(Kernels, BinaryFwd_Mul) {
    auto c = run_binary_fwd({2, 3, 4}, {5, 6, 7}, 3, 1, MulBinary{});
    EXPECT_NEAR(c(0), 10.0f, 1e-5f);
    EXPECT_NEAR(c(1), 18.0f, 1e-5f);
    EXPECT_NEAR(c(2), 28.0f, 1e-5f);
}

TEST(Kernels, BinaryFwd_Div) {
    auto c = run_binary_fwd({6.0f, 9.0f}, {2.0f, 3.0f}, 2, 1, DivBinary{});
    EXPECT_NEAR(c(0), 3.0f, 1e-5f);
    EXPECT_NEAR(c(1), 3.0f, 1e-5f);
}

// ElemwiseBinary backward

static std::pair<CPUMatrix<float>, CPUMatrix<float>> run_binary_bwd(
    std::initializer_list<float> a_vals,
    std::initializer_list<float> b_vals,
    std::initializer_list<float> g_vals,
    int rows,
    int cols,
    const ElemwiseBinary::Op& op
) {
    CPUMatrix<float> ca(Shape(rows, cols)), cb(Shape(rows, cols)), cg(Shape(rows, cols));
    int i = 0;
    for (float v : a_vals)
        ca(i++, 0) = v;
    i = 0;
    for (float v : b_vals)
        cb(i++, 0) = v;
    i = 0;
    for (float v : g_vals)
        cg(i++, 0) = v;

    auto ga = GPUMatrix<float>::from_cpu(ca);
    auto gb = GPUMatrix<float>::from_cpu(cb);
    auto gg = GPUMatrix<float>::from_cpu(cg);
    GPUMatrix<float> ga_g(Shape(rows, cols));
    ga_g.clear();
    GPUMatrix<float> gb_g(Shape(rows, cols));
    gb_g.clear();

    ElemwiseBinary::backward(ga, ga_g, gb, gb_g, gg, op);
    cudaDeviceSynchronize();
    return {ga_g.to_cpu(), gb_g.to_cpu()};
}

TEST(Kernels, BinaryBwd_Add) {
    auto [ga, gb] = run_binary_bwd({1}, {2}, {3}, 1, 1, AddBinary{});
    EXPECT_NEAR(ga(0), 3.0f, 1e-5f);
    EXPECT_NEAR(gb(0), 3.0f, 1e-5f);
}

TEST(Kernels, BinaryBwd_Sub) {
    auto [ga, gb] = run_binary_bwd({5}, {2}, {2}, 1, 1, SubBinary{});
    EXPECT_NEAR(ga(0), 2.0f, 1e-5f);
    EXPECT_NEAR(gb(0), -2.0f, 1e-5f);
}

TEST(Kernels, BinaryBwd_Mul) {
    auto [ga, gb] = run_binary_bwd({3}, {4}, {1}, 1, 1, MulBinary{});
    EXPECT_NEAR(ga(0), 4.0f, 1e-5f);
    EXPECT_NEAR(gb(0), 3.0f, 1e-5f);
}

TEST(Kernels, BinaryBwd_Div) {
    auto [ga, gb] = run_binary_bwd({6}, {3}, {1}, 1, 1, DivBinary{});
    EXPECT_NEAR(ga(0), 1.0f / 3.0f, 1e-5f);
    EXPECT_NEAR(gb(0), -6.0f / (3.0f * 3.0f), 1e-5f);
}

// ElemwiseBinary broadcast forward / backward

TEST(Kernels, BinaryBroadcast_Fwd_Add) {
    CPUMatrix<float> bias(Shape(4, 1));
    CPUMatrix<float> data(Shape(4, 3));
    for (int r = 0; r < 4; ++r) {
        bias(r, 0) = (float)(r + 1);
        for (int c = 0; c < 3; ++c)
            data(r, c) = 10.0f;
    }
    auto gb = GPUMatrix<float>::from_cpu(bias);
    auto gd = GPUMatrix<float>::from_cpu(data);
    GPUMatrix<float> gout(Shape(4, 3));

    ElemwiseBinary::broadcast_forward(gb, gd, gout, AddBinary{});
    cudaDeviceSynchronize();
    auto cpu = gout.to_cpu();

    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 4; ++r)
            EXPECT_NEAR(cpu(r, c), 10.0f + (float)(r + 1), 1e-5f);
}

TEST(Kernels, BinaryBroadcast_Bwd_Add_BiasGrad) {
    CPUMatrix<float> bias(Shape(4, 1));
    CPUMatrix<float> data(Shape(4, 3));
    CPUMatrix<float> out_g(Shape(4, 3));
    bias.fill(0.0f);
    data.fill(1.0f);
    out_g.fill(1.0f);

    auto gb = GPUMatrix<float>::from_cpu(bias);
    auto gd = GPUMatrix<float>::from_cpu(data);
    auto go = GPUMatrix<float>::from_cpu(out_g);
    GPUMatrix<float> gb_g(Shape(4, 1));
    gb_g.clear();
    GPUMatrix<float> gd_g(Shape(4, 3));
    gd_g.clear();

    ElemwiseBinary::broadcast_backward(gb, gb_g, gd, gd_g, go, AddBinary{});
    cudaDeviceSynchronize();
    auto bias_grad = gb_g.to_cpu();
    auto data_grad = gd_g.to_cpu();

    for (int r = 0; r < 4; ++r)
        EXPECT_NEAR(bias_grad(r, 0), 3.0f, 1e-4f);

    for (int i = 0; i < 12; ++i)
        EXPECT_NEAR(data_grad(i), 1.0f, 1e-5f);
}

// MatMul forward / backward

TEST(Kernels, MatMul_Forward) {
    CPUMatrix<float> w(Shape(2, 3));
    w.fill(1.0f);
    CPUMatrix<float> x(Shape(3, 4));
    x.fill(1.0f);
    auto gw = GPUMatrix<float>::from_cpu(w);
    auto gx = GPUMatrix<float>::from_cpu(x);
    GPUMatrix<float> gy(Shape(2, 4));

    MatMul::forward(gw, gx, gy);
    cudaDeviceSynchronize();
    auto cpu = gy.to_cpu();

    for (int i = 0; i < 8; ++i)
        EXPECT_NEAR(cpu(i), 3.0f, 1e-4f);
}

TEST(Kernels, MatMul_Backward_WeightGrad) {
    CPUMatrix<float> w(Shape(2, 3));
    w.fill(0.0f);
    CPUMatrix<float> x(Shape(3, 4));
    x.fill(1.0f);
    CPUMatrix<float> dy(Shape(2, 4));
    dy.fill(1.0f);

    auto gw = GPUMatrix<float>::from_cpu(w);
    auto gx = GPUMatrix<float>::from_cpu(x);
    auto gdy = GPUMatrix<float>::from_cpu(dy);
    GPUMatrix<float> gw_g(Shape(2, 3));
    gw_g.clear();
    GPUMatrix<float> gx_g(Shape(3, 4));
    gx_g.clear();

    MatMul::backward(gw, gw_g, gx, gx_g, gdy);
    cudaDeviceSynchronize();

    auto dw = gw_g.to_cpu();
    auto dx = gx_g.to_cpu();

    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(dw(i), 4.0f, 1e-4f);

    for (int i = 0; i < 12; ++i)
        EXPECT_NEAR(dx(i), 0.0f, 1e-4f);
}

TEST(Kernels, MatMul_Backward_InputGrad) {
    CPUMatrix<float> w(Shape(2, 3));
    w.fill(1.0f);
    CPUMatrix<float> x(Shape(3, 2));
    x.fill(0.0f);
    CPUMatrix<float> dy(Shape(2, 2));
    dy.fill(1.0f);

    auto gw = GPUMatrix<float>::from_cpu(w);
    auto gx = GPUMatrix<float>::from_cpu(x);
    auto gdy = GPUMatrix<float>::from_cpu(dy);
    GPUMatrix<float> gw_g(Shape(2, 3));
    gw_g.clear();
    GPUMatrix<float> gx_g(Shape(3, 2));
    gx_g.clear();

    MatMul::backward(gw, gw_g, gx, gx_g, gdy);
    cudaDeviceSynchronize();
    auto dx = gx_g.to_cpu();

    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(dx(i), 2.0f, 1e-4f);
}
