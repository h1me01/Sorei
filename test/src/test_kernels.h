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

    ElemwiseUnary::backward(dev_in, dev_in_g, dev_og, op, false);
    cudaDeviceSynchronize();
    return dev_in_g.to_host();
}

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

TEST(Kernels, UnaryFwd_Identity) {
    auto out = run_unary_fwd({-1.0f, 0.0f, 2.5f}, unary::Identity{});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), -1.0f, 1e-6f);
    EXPECT_NEAR(host(1), 0.0f, 1e-6f);
    EXPECT_NEAR(host(2), 2.5f, 1e-6f);
}

TEST(Kernels, UnaryFwd_ReLU) {
    auto out = run_unary_fwd({-2.0f, 0.0f, 1.5f}, unary::ReLU{});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), 0.0f, 1e-6f);
    EXPECT_NEAR(host(1), 0.0f, 1e-6f);
    EXPECT_NEAR(host(2), 1.5f, 1e-6f);
}

TEST(Kernels, UnaryFwd_ClampedReLU) {
    auto out = run_unary_fwd({-1.0f, 0.5f, 2.0f}, unary::ClampedReLU{});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), 0.0f, 1e-6f);
    EXPECT_NEAR(host(1), 0.5f, 1e-6f);
    EXPECT_NEAR(host(2), 1.0f, 1e-6f);
}

TEST(Kernels, UnaryFwd_SquaredClampedReLU) {
    auto out = run_unary_fwd({-1.0f, 0.5f, 2.0f}, unary::SquaredClampedReLU{});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), 0.0f, 1e-6f);
    EXPECT_NEAR(host(1), 0.25f, 1e-5f);
    EXPECT_NEAR(host(2), 1.0f, 1e-6f);
}

TEST(Kernels, UnaryFwd_Sigmoid) {
    auto out = run_unary_fwd({0.0f}, unary::Sigmoid{});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), 0.5f, 1e-5f);

    auto out2 = run_unary_fwd({10.0f, -10.0f}, unary::Sigmoid{});
    auto host2 = out2.to_host();
    EXPECT_GT((double)host2(0), 0.99);
    EXPECT_LT((double)host2(1), 0.01);
}

TEST(Kernels, UnaryFwd_Abs) {
    auto out = run_unary_fwd({-3.0f, 0.0f, 2.0f}, unary::Abs{});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), 3.0f, 1e-6f);
    EXPECT_NEAR(host(1), 0.0f, 1e-6f);
    EXPECT_NEAR(host(2), 2.0f, 1e-6f);
}

TEST(Kernels, UnaryFwd_Clamp) {
    auto out = run_unary_fwd({-5.0f, 0.3f, 5.0f}, unary::Clamp{-1.0f, 1.0f});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), -1.0f, 1e-6f);
    EXPECT_NEAR(host(1), 0.3f, 1e-6f);
    EXPECT_NEAR(host(2), 1.0f, 1e-6f);
}

TEST(Kernels, UnaryFwd_AddScale) {
    auto out = run_unary_fwd({1.0f, -2.0f}, unary::AddScale{2.0f, 3.0f});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), 5.0f, 1e-5f);
    EXPECT_NEAR(host(1), -1.0f, 1e-5f);
}

TEST(Kernels, UnaryFwd_DivLeft) {
    auto out = run_unary_fwd({2.0f, 4.0f}, unary::DivLeft{8.0f});
    auto host = out.to_host();
    EXPECT_NEAR(host(0), 4.0f, 1e-5f);
    EXPECT_NEAR(host(1), 2.0f, 1e-5f);
}

TEST(Kernels, UnaryBwd_ReLU_Positive) {
    auto g = run_unary_bwd({2.0f}, {3.0f}, unary::ReLU{});
    EXPECT_NEAR(g(0), 3.0f, 1e-5f);
}

TEST(Kernels, UnaryBwd_ReLU_Negative) {
    auto g = run_unary_bwd({-1.0f}, {3.0f}, unary::ReLU{});
    EXPECT_NEAR(g(0), 0.0f, 1e-5f);
}

TEST(Kernels, UnaryBwd_Sigmoid) {
    auto g = run_unary_bwd({0.0f}, {2.0f}, unary::Sigmoid{});
    EXPECT_NEAR(g(0), 0.5f, 1e-4f);
}

TEST(Kernels, UnaryBwd_Abs) {
    auto g_pos = run_unary_bwd({3.0f}, {1.0f}, unary::Abs{});
    auto g_neg = run_unary_bwd({-2.0f}, {1.0f}, unary::Abs{});
    EXPECT_NEAR(g_pos(0), 1.0f, 1e-5f);
    EXPECT_NEAR(g_neg(0), -1.0f, 1e-5f);
}

TEST(Kernels, UnaryBwd_AddScale) {
    auto g = run_unary_bwd({5.0f}, {1.0f}, unary::AddScale{4.0f, 0.0f});
    EXPECT_NEAR(g(0), 4.0f, 1e-5f);
}

TEST(Kernels, UnaryBwd_ClampedReLU) {
    auto g_in = run_unary_bwd({0.5f}, {2.0f}, unary::ClampedReLU{});
    EXPECT_NEAR(g_in(0), 2.0f, 1e-5f);
    auto g_lo = run_unary_bwd({-0.5f}, {2.0f}, unary::ClampedReLU{});
    EXPECT_NEAR(g_lo(0), 0.0f, 1e-5f);
    auto g_hi = run_unary_bwd({1.5f}, {2.0f}, unary::ClampedReLU{});
    EXPECT_NEAR(g_hi(0), 0.0f, 1e-5f);
}

TEST(Kernels, UnaryBwd_SquaredClampedReLU) {
    auto g_in = run_unary_bwd({0.5f}, {1.0f}, unary::SquaredClampedReLU{});
    EXPECT_NEAR(g_in(0), 1.0f, 1e-5f);
    auto g_lo = run_unary_bwd({-1.0f}, {1.0f}, unary::SquaredClampedReLU{});
    EXPECT_NEAR(g_lo(0), 0.0f, 1e-5f);
    auto g_hi = run_unary_bwd({2.0f}, {1.0f}, unary::SquaredClampedReLU{});
    EXPECT_NEAR(g_hi(0), 0.0f, 1e-5f);
}

TEST(Kernels, UnaryBwd_Clamp) {
    auto g_in = run_unary_bwd({0.5f}, {2.0f}, unary::Clamp{-1.0f, 1.0f});
    EXPECT_NEAR(g_in(0), 2.0f, 1e-5f);
    auto g_hi = run_unary_bwd({2.0f}, {2.0f}, unary::Clamp{-1.0f, 1.0f});
    EXPECT_NEAR(g_hi(0), 0.0f, 1e-5f);
    auto g_lo = run_unary_bwd({-2.0f}, {2.0f}, unary::Clamp{-1.0f, 1.0f});
    EXPECT_NEAR(g_lo(0), 0.0f, 1e-5f);
}

TEST(Kernels, UnaryBwd_DivLeft) {
    auto g0 = run_unary_bwd({2.0f}, {1.0f}, unary::DivLeft{6.0f});
    EXPECT_NEAR(g0(0), -1.5f, 1e-4f);
    auto g1 = run_unary_bwd({3.0f}, {1.0f}, unary::DivLeft{6.0f});
    EXPECT_NEAR(g1(0), -6.0f / 9.0f, 1e-4f);
}

TEST(Kernels, BinaryFwd_Add) {
    auto c = run_binary_fwd({1, 2, 3}, {4, 5, 6}, 3, 1, binary::Add{});
    EXPECT_NEAR(c(0), 5.0f, 1e-5f);
    EXPECT_NEAR(c(1), 7.0f, 1e-5f);
    EXPECT_NEAR(c(2), 9.0f, 1e-5f);
}

TEST(Kernels, BinaryFwd_Sub) {
    auto c = run_binary_fwd({5, 5, 5}, {1, 2, 3}, 3, 1, binary::Sub{});
    EXPECT_NEAR(c(0), 4.0f, 1e-5f);
    EXPECT_NEAR(c(1), 3.0f, 1e-5f);
    EXPECT_NEAR(c(2), 2.0f, 1e-5f);
}

TEST(Kernels, BinaryFwd_Mul) {
    auto c = run_binary_fwd({2, 3, 4}, {5, 6, 7}, 3, 1, binary::Mul{});
    EXPECT_NEAR(c(0), 10.0f, 1e-5f);
    EXPECT_NEAR(c(1), 18.0f, 1e-5f);
    EXPECT_NEAR(c(2), 28.0f, 1e-5f);
}

TEST(Kernels, BinaryFwd_Div) {
    auto c = run_binary_fwd({6.0f, 9.0f}, {2.0f, 3.0f}, 2, 1, binary::Div{});
    EXPECT_NEAR(c(0), 3.0f, 1e-5f);
    EXPECT_NEAR(c(1), 3.0f, 1e-5f);
}

TEST(Kernels, BinaryBwd_Add) {
    auto [ga, gb] = run_binary_bwd({1}, {2}, {3}, 1, 1, binary::Add{});
    EXPECT_NEAR(ga(0), 3.0f, 1e-5f);
    EXPECT_NEAR(gb(0), 3.0f, 1e-5f);
}

TEST(Kernels, BinaryBwd_Sub) {
    auto [ga, gb] = run_binary_bwd({5}, {2}, {2}, 1, 1, binary::Sub{});
    EXPECT_NEAR(ga(0), 2.0f, 1e-5f);
    EXPECT_NEAR(gb(0), -2.0f, 1e-5f);
}

TEST(Kernels, BinaryBwd_Mul) {
    auto [ga, gb] = run_binary_bwd({3}, {4}, {1}, 1, 1, binary::Mul{});
    EXPECT_NEAR(ga(0), 4.0f, 1e-5f);
    EXPECT_NEAR(gb(0), 3.0f, 1e-5f);
}

TEST(Kernels, BinaryBwd_Div) {
    auto [ga, gb] = run_binary_bwd({6}, {3}, {1}, 1, 1, binary::Div{});
    EXPECT_NEAR(ga(0), 1.0f / 3.0f, 1e-5f);
    EXPECT_NEAR(gb(0), -6.0f / (3.0f * 3.0f), 1e-5f);
}

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

    ElemwiseBinary::broadcast_forward(gb, gd, gout, binary::Add{});
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

    ElemwiseBinary::broadcast_backward(gb, gb_g, gd, gd_g, go, binary::Add{});
    cudaDeviceSynchronize();
    auto bias_grad = gb_g.to_host();
    auto data_grad = gd_g.to_host();

    for (int r = 0; r < 4; ++r)
        EXPECT_NEAR(bias_grad(r, 0), 3.0f, 1e-4f);

    for (int i = 0; i < 12; ++i)
        EXPECT_NEAR(data_grad(i), 1.0f, 1e-5f);
}

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
