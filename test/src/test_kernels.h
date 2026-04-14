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
    DeviceMatrix<float> m({4, 3});
    set(m, 7.5f);
    cudaDeviceSynchronize();
    auto host = m.to_host();
    for (int i = 0; i < 12; ++i)
        EXPECT_NEAR(host(i), 7.5f, 1e-6f);
}

TEST(Kernels, Set_ZeroFill) {
    DeviceMatrix<float> m({5, 5});
    set(m, 1.0f);
    set(m, 0.0f);
    cudaDeviceSynchronize();
    auto host = m.to_host();
    for (int i = 0; i < 25; ++i)
        EXPECT_EQ(host(i), 0.0f);
}

// cublas::sgemm

// build a DeviceMatrix from a flat row-major initializer
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

TEST(Kernels, SGemm_AlphaBeta) {
    auto A = make_device({2, 2}, {1, 0, 0, 1});
    auto B = make_device({2, 2}, {3, 4, 5, 6});
    DeviceMatrix<float> C({2, 2});

    cublas::sgemm(false, false, 1.0f, A, B, 0.0f, C);
    cudaDeviceSynchronize();

    cublas::sgemm(false, false, 2.0f, A, B, 0.5f, C);
    cudaDeviceSynchronize();
    auto host = C.to_host();

    EXPECT_NEAR(host(0, 0), 7.5f, 1e-3f);
    EXPECT_NEAR(host(1, 0), 12.5f, 1e-3f);
    EXPECT_NEAR(host(0, 1), 10.0f, 1e-3f);
    EXPECT_NEAR(host(1, 1), 15.0f, 1e-3f);
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

// ElemwiseUnary

static DeviceMatrix<float>
run_unary_fwd(std::initializer_list<float> vals, const ElemwiseUnary::Op& op) {
    int n = vals.size();
    HostMatrix<float> host_in({n, 1});
    int i = 0;
    for (float v : vals)
        host_in(i++, 0) = v;

    auto dev_in = DeviceMatrix<float>::from_host(host_in);
    DeviceMatrix<float> dev_out({n, 1});
    ElemwiseUnary::forward(dev_in, dev_out, op);
    cudaDeviceSynchronize();

    return dev_out;
}

TEST(Kernels, UnaryFwd_Identity) {
    auto out = run_unary_fwd({-1.0f, 0.0f, 2.5f}, Identity{});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), -1.0f, 1e-6f);
    EXPECT_NEAR(host(1), 0.0f, 1e-6f);
    EXPECT_NEAR(host(2), 2.5f, 1e-6f);
}

TEST(Kernels, UnaryFwd_ReLU) {
    auto out = run_unary_fwd({-2.0f, 0.0f, 1.5f}, ReLU{});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), 0.0f, 1e-6f);
    EXPECT_NEAR(host(1), 0.0f, 1e-6f);
    EXPECT_NEAR(host(2), 1.5f, 1e-6f);
}

TEST(Kernels, UnaryFwd_ClampedReLU) {
    auto out = run_unary_fwd({-1.0f, 0.5f, 2.0f}, ClampedReLU{});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), 0.0f, 1e-6f);
    EXPECT_NEAR(host(1), 0.5f, 1e-6f);
    EXPECT_NEAR(host(2), 1.0f, 1e-6f);
}

TEST(Kernels, UnaryFwd_SquaredClampedReLU) {
    auto out = run_unary_fwd({-1.0f, 0.5f, 2.0f}, SquaredClampedReLU{});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), 0.0f, 1e-6f);
    EXPECT_NEAR(host(1), 0.25f, 1e-5f);
    EXPECT_NEAR(host(2), 1.0f, 1e-6f);
}

TEST(Kernels, UnaryFwd_Sigmoid) {
    auto out = run_unary_fwd({0.0f}, Sigmoid{});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), 0.5f, 1e-5f);

    auto out2 = run_unary_fwd({10.0f, -10.0f}, Sigmoid{});
    auto host2 = out2.to_host();
    EXPECT_GT((double)host2(0), 0.99);
    EXPECT_LT((double)host2(1), 0.01);
}

TEST(Kernels, UnaryFwd_Abs) {
    auto out = run_unary_fwd({-3.0f, 0.0f, 2.0f}, Abs{});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), 3.0f, 1e-6f);
    EXPECT_NEAR(host(1), 0.0f, 1e-6f);
    EXPECT_NEAR(host(2), 2.0f, 1e-6f);
}

TEST(Kernels, UnaryFwd_PowInt) {
    auto out = run_unary_fwd({2.0f, -3.0f}, PowInt{3});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), 8.0f, 1e-4f);
    EXPECT_NEAR(host(1), -27.0f, 1e-4f);
}

TEST(Kernels, UnaryFwd_PowFloat) {
    auto out = run_unary_fwd({4.0f, 9.0f}, PowFloat{0.5f});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), 2.0f, 1e-4f);
    EXPECT_NEAR(host(1), 3.0f, 1e-4f);
}

TEST(Kernels, UnaryFwd_Clamp) {
    auto out = run_unary_fwd({-5.0f, 0.3f, 5.0f}, Clamp{-1.0f, 1.0f});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), -1.0f, 1e-6f);
    EXPECT_NEAR(host(1), 0.3f, 1e-6f);
    EXPECT_NEAR(host(2), 1.0f, 1e-6f);
}

TEST(Kernels, UnaryFwd_AddScaleUnary) {
    auto out = run_unary_fwd({1.0f, -2.0f}, AddScaleUnary{2.0f, 3.0f});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), 5.0f, 1e-5f);
    EXPECT_NEAR(host(1), -1.0f, 1e-5f);
}

TEST(Kernels, UnaryFwd_DivLeftUnary) {
    auto out = run_unary_fwd({2.0f, 4.0f}, DivLeftUnary{8.0f});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), 4.0f, 1e-5f);
    EXPECT_NEAR(host(1), 2.0f, 1e-5f);
}

// ElemwiseUnary backward

static HostMatrix<float> run_unary_bwd(
    std::initializer_list<float> inputs,
    std::initializer_list<float> out_grads,
    const ElemwiseUnary::Op& op
) {
    int n = inputs.size();
    HostMatrix<float> host_in({n, 1});
    HostMatrix<float> host_og({n, 1});
    int i = 0;
    for (float v : inputs)
        host_in(i++, 0) = v;
    i = 0;
    for (float v : out_grads)
        host_og(i++, 0) = v;

    auto dev_in = DeviceMatrix<float>::from_host(host_in);
    DeviceMatrix<float> dev_in_g({n, 1});
    dev_in_g.clear();
    auto dev_og = DeviceMatrix<float>::from_host(host_og);

    ElemwiseUnary::backward(dev_in, dev_in_g, dev_og, op);
    cudaDeviceSynchronize();
    return dev_in_g.to_host();
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

static HostMatrix<float> run_binary_fwd(
    std::initializer_list<float> a_vals,
    std::initializer_list<float> b_vals,
    int rows,
    int cols,
    const ElemwiseBinary::Op& op
) {
    HostMatrix<float> ca({rows, cols}), cb({rows, cols});
    int i = 0;
    for (float v : a_vals)
        ca(i++, 0) = v;
    i = 0;
    for (float v : b_vals)
        cb(i++, 0) = v;

    auto ga = DeviceMatrix<float>::from_host(ca);
    auto gb = DeviceMatrix<float>::from_host(cb);
    DeviceMatrix<float> gc({rows, cols});

    ElemwiseBinary::forward(ga, gb, gc, op);
    cudaDeviceSynchronize();
    return gc.to_host();
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

static std::pair<HostMatrix<float>, HostMatrix<float>> run_binary_bwd(
    std::initializer_list<float> a_vals,
    std::initializer_list<float> b_vals,
    std::initializer_list<float> g_vals,
    int rows,
    int cols,
    const ElemwiseBinary::Op& op
) {
    HostMatrix<float> ca({rows, cols}), cb({rows, cols}), cg({rows, cols});
    int i = 0;
    for (float v : a_vals)
        ca(i++, 0) = v;
    i = 0;
    for (float v : b_vals)
        cb(i++, 0) = v;
    i = 0;
    for (float v : g_vals)
        cg(i++, 0) = v;

    auto ga = DeviceMatrix<float>::from_host(ca);
    auto gb = DeviceMatrix<float>::from_host(cb);
    auto gg = DeviceMatrix<float>::from_host(cg);
    DeviceMatrix<float> ga_g({rows, cols});
    ga_g.clear();
    DeviceMatrix<float> gb_g({rows, cols});
    gb_g.clear();

    ElemwiseBinary::backward(ga, ga_g, gb, gb_g, gg, op);
    cudaDeviceSynchronize();
    return {ga_g.to_host(), gb_g.to_host()};
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
    HostMatrix<float> bias({4, 1});
    HostMatrix<float> data({4, 3});
    for (int r = 0; r < 4; ++r) {
        bias(r, 0) = (float)(r + 1);
        for (int c = 0; c < 3; ++c)
            data(r, c) = 10.0f;
    }
    auto gb = DeviceMatrix<float>::from_host(bias);
    auto gd = DeviceMatrix<float>::from_host(data);
    DeviceMatrix<float> gout({4, 3});

    ElemwiseBinary::broadcast_forward(gb, gd, gout, AddBinary{});
    cudaDeviceSynchronize();
    auto host = gout.to_host();

    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 4; ++r)
            EXPECT_NEAR(host(r, c), 10.0f + (float)(r + 1), 1e-5f);
}

TEST(Kernels, BinaryBroadcast_Bwd_Add_BiasGrad) {
    HostMatrix<float> bias({4, 1});
    HostMatrix<float> data({4, 3});
    HostMatrix<float> out_g({4, 3});
    bias.fill(0.0f);
    data.fill(1.0f);
    out_g.fill(1.0f);

    auto gb = DeviceMatrix<float>::from_host(bias);
    auto gd = DeviceMatrix<float>::from_host(data);
    auto go = DeviceMatrix<float>::from_host(out_g);
    DeviceMatrix<float> gb_g({4, 1});
    gb_g.clear();
    DeviceMatrix<float> gd_g({4, 3});
    gd_g.clear();

    ElemwiseBinary::broadcast_backward(gb, gb_g, gd, gd_g, go, AddBinary{});
    cudaDeviceSynchronize();
    auto bias_grad = gb_g.to_host();
    auto data_grad = gd_g.to_host();

    for (int r = 0; r < 4; ++r)
        EXPECT_NEAR(bias_grad(r, 0), 3.0f, 1e-4f);

    for (int i = 0; i < 12; ++i)
        EXPECT_NEAR(data_grad(i), 1.0f, 1e-5f);
}

// MatMul forward / backward

TEST(Kernels, MatMul_Forward) {
    HostMatrix<float> w({2, 3});
    w.fill(1.0f);
    HostMatrix<float> x({3, 4});
    x.fill(1.0f);
    auto gw = DeviceMatrix<float>::from_host(w);
    auto gx = DeviceMatrix<float>::from_host(x);
    DeviceMatrix<float> gy({2, 4});

    MatMul::forward(gw, gx, gy);
    cudaDeviceSynchronize();
    auto host = gy.to_host();

    for (int i = 0; i < 8; ++i)
        EXPECT_NEAR(host(i), 3.0f, 1e-4f);
}

TEST(Kernels, MatMul_Backward_WeightGrad) {
    HostMatrix<float> w({2, 3});
    w.fill(0.0f);
    HostMatrix<float> x({3, 4});
    x.fill(1.0f);
    HostMatrix<float> dy({2, 4});
    dy.fill(1.0f);

    auto gw = DeviceMatrix<float>::from_host(w);
    auto gx = DeviceMatrix<float>::from_host(x);
    auto gdy = DeviceMatrix<float>::from_host(dy);
    DeviceMatrix<float> gw_g({2, 3});
    gw_g.clear();
    DeviceMatrix<float> gx_g({3, 4});
    gx_g.clear();

    MatMul::backward(gw, gw_g, gx, gx_g, gdy);
    cudaDeviceSynchronize();

    auto dw = gw_g.to_host();
    auto dx = gx_g.to_host();

    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(dw(i), 4.0f, 1e-4f);

    for (int i = 0; i < 12; ++i)
        EXPECT_NEAR(dx(i), 0.0f, 1e-4f);
}

TEST(Kernels, MatMul_Backward_InputGrad) {
    HostMatrix<float> w({2, 3});
    w.fill(1.0f);
    HostMatrix<float> x({3, 2});
    x.fill(0.0f);
    HostMatrix<float> dy({2, 2});
    dy.fill(1.0f);

    auto gw = DeviceMatrix<float>::from_host(w);
    auto gx = DeviceMatrix<float>::from_host(x);
    auto gdy = DeviceMatrix<float>::from_host(dy);
    DeviceMatrix<float> gw_g({2, 3});
    gw_g.clear();
    DeviceMatrix<float> gx_g({3, 2});
    gx_g.clear();

    MatMul::backward(gw, gw_g, gx, gx_g, gdy);
    cudaDeviceSynchronize();
    auto dx = gx_g.to_host();

    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(dx(i), 2.0f, 1e-4f);
}
