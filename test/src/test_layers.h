#pragma once

#include <cuda_runtime.h>

#include "sorei/nn.h"

#include "framework.h"
#include "grad_check.h"

using namespace sorei;
using namespace sorei::matrix;
using namespace sorei::nn;
using namespace sorei::cuda;

TEST(Layer, Param_Shape) {
    Param p({8, 4});
    EXPECT_EQ(p.data().rows(), 8);
    EXPECT_EQ(p.data().cols(), 4);
}

TEST(Layer, InputFloat_Shape) {
    InputFloat inp({5, 3});
    EXPECT_EQ(inp.data().rows(), 5);
    EXPECT_EQ(inp.data().cols(), 3);
}

TEST(Layer, ElemwiseUnary_Shape) {
    Param p({6, 4});
    ElemwiseUnary layer(&p, ReLU{});
    EXPECT_EQ(layer.data().rows(), 6);
    EXPECT_EQ(layer.data().cols(), 4);
}

TEST(Layer, ElemwiseBinary_Shape_NonBroadcast) {
    Param a({4, 3});
    Param b({4, 3});
    ElemwiseBinary layer(&a, &b, AddBinary{});
    EXPECT_EQ(layer.data().rows(), 4);
    EXPECT_EQ(layer.data().cols(), 3);
}

TEST(Layer, ElemwiseBinary_Shape_Broadcast) {
    Param bias({4, 1});
    Param data({4, 5});
    ElemwiseBinary layer(&bias, &data, AddBinary{});
    EXPECT_EQ(layer.data().rows(), 4);
    EXPECT_EQ(layer.data().cols(), 5);
}

TEST(Layer, MatMul_Shape) {
    Param w({8, 6});
    Param x({6, 4});
    MatMul layer(&w, &x);
    EXPECT_EQ(layer.data().rows(), 8);
    EXPECT_EQ(layer.data().cols(), 4);
}

TEST(Layer, Affine_Shape) {
    Param w({8, 6});
    Param x({6, 4});
    Param b({8, 1});
    Affine layer(&x, &w, &b);
    EXPECT_EQ(layer.data().rows(), 8);
    EXPECT_EQ(layer.data().cols(), 4);
}

TEST(Layer, Mean_Shape) {
    Param p({5, 8});
    Mean layer(&p);
    EXPECT_EQ(layer.data().rows(), 1);
    EXPECT_EQ(layer.data().cols(), 1);
}

TEST(Layer, PairwiseMul_Shape) {
    Param p({8, 3});
    PairwiseMul layer(&p);
    EXPECT_EQ(layer.data().rows(), 4);
    EXPECT_EQ(layer.data().cols(), 3);
}

TEST(Layer, Concat_Rows_Shape) {
    Param a({3, 4});
    Param b({5, 4});
    Concat layer({&a, &b}, ConcatAxis::Rows);
    EXPECT_EQ(layer.data().rows(), 8);
    EXPECT_EQ(layer.data().cols(), 4);
}

TEST(Layer, Concat_Cols_Shape) {
    Param a({4, 2});
    Param b({4, 3});
    Concat layer({&a, &b}, ConcatAxis::Cols);
    EXPECT_EQ(layer.data().rows(), 4);
    EXPECT_EQ(layer.data().cols(), 5);
}

TEST(Layer, SoftmaxCrossEntropy_Shape) {
    Param logits({10, 5});
    Input<int> labels({1, 5});
    SoftmaxCrossEntropy layer(&logits, &labels);
    EXPECT_EQ(layer.data().rows(), 1);
    EXPECT_EQ(layer.data().cols(), 5);
}

TEST(Layer, BucketIndex_Shape) {
    BucketIndex bidx(3, 8);
    EXPECT_EQ(bidx.data().rows(), 1);
    EXPECT_EQ(bidx.data().cols(), 8);
}

TEST(Layer, Select_Shape) {
    const int count = 4, out_dim = 8, batch = 5;
    Param inp({count * out_dim, batch});
    BucketIndex bidx(count, batch);
    Select sel(&inp, &bidx);
    EXPECT_EQ(sel.data().rows(), out_dim);
    EXPECT_EQ(sel.data().cols(), batch);
}

TEST(Layer, SparseAffine_Shape) {
    const int out_dim = 4, n_features = 8, max_entries = 5, batch = 6;
    Param weight({out_dim, n_features});
    Param bias({out_dim, 1});
    Input<int> indices({max_entries, batch});
    SparseAffine layer(&indices, &weight, &bias);
    EXPECT_EQ(layer.data().rows(), out_dim);
    EXPECT_EQ(layer.data().cols(), batch);
}

TEST(Layer, ElemwiseUnary_Forward_ReLU) {
    auto p = test::make_param_filled({4, 1}, -1.0f);
    ElemwiseUnary layer(p.get(), ReLU{});

    layer.forward();
    cudaDeviceSynchronize();

    auto host = layer.data().to_host();
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(host(i), 0.0f, 1e-6f);
}

TEST(Layer, ElemwiseBinary_Forward_Add) {
    auto a = test::make_param_filled({3, 2}, 1.0f);
    auto b = test::make_param_filled({3, 2}, 2.0f);
    ElemwiseBinary layer(a.get(), b.get(), AddBinary{});

    layer.forward();
    cudaDeviceSynchronize();

    auto host = layer.data().to_host();
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(host(i), 3.0f, 1e-5f);
}

TEST(Layer, MatMul_Forward) {
    auto w = test::make_param_filled({2, 3}, 1.0f);
    auto x = test::make_param_filled({3, 4}, 1.0f);

    MatMul layer(w.get(), x.get());
    layer.forward();
    cudaDeviceSynchronize();

    auto host = layer.data().to_host();
    for (int i = 0; i < 8; ++i)
        EXPECT_NEAR(host(i), 3.0f, 1e-4f);
}

TEST(Layer, Mean_Forward) {
    HostMatrix<float> src({3, 4});
    for (int i = 0; i < 12; ++i)
        src(i) = (float)i;

    Param p({3, 4});
    p.data().upload(src);
    p.grad().clear();

    Mean layer(&p);
    layer.forward();
    cudaDeviceSynchronize();
    EXPECT_NEAR(layer.data().to_host()(0), 5.5f, 1e-4f);
}

TEST(Layer, PairwiseMul_Forward) {
    HostMatrix<float> src({4, 1});
    src(0, 0) = 1;
    src(1, 0) = 2;
    src(2, 0) = 3;
    src(3, 0) = 4;

    Param p({4, 1});
    p.data().upload(src);
    p.grad().clear();

    PairwiseMul layer(&p);
    layer.forward();
    cudaDeviceSynchronize();

    auto host = layer.data().to_host();
    EXPECT_NEAR(host(0, 0), 3.0f, 1e-5f);
    EXPECT_NEAR(host(1, 0), 8.0f, 1e-5f);
}

TEST(Layer, Affine_Forward) {
    auto w = test::make_param_filled({2, 3}, 1.0f);
    auto x = test::make_param_filled({3, 2}, 1.0f);

    HostMatrix<float> bias_src({2, 1});
    bias_src(0, 0) = 1.0f;
    bias_src(1, 0) = 2.0f;
    Param b({2, 1});
    b.data().upload(bias_src);
    b.grad().clear();

    Affine layer(x.get(), w.get(), &b);
    layer.forward();
    cudaDeviceSynchronize();

    auto host = layer.data().to_host();
    for (int c = 0; c < 2; ++c) {
        EXPECT_NEAR(host(0, c), 4.0f, 1e-4f);
        EXPECT_NEAR(host(1, c), 5.0f, 1e-4f);
    }
}

TEST(Layer, Concat_Rows_Forward) {
    auto a = test::make_param_filled({2, 3}, 1.0f);
    auto b = test::make_param_filled({4, 3}, 2.0f);
    Concat layer({a.get(), b.get()}, ConcatAxis::Rows);

    layer.forward();
    cudaDeviceSynchronize();

    auto host = layer.data().to_host();
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < 2; ++r)
            EXPECT_NEAR(host(r, c), 1.0f, 1e-5f);
        for (int r = 2; r < 6; ++r)
            EXPECT_NEAR(host(r, c), 2.0f, 1e-5f);
    }
}

TEST(Layer, Concat_Cols_Forward) {
    auto a = test::make_param_filled({3, 2}, 1.0f);
    auto b = test::make_param_filled({3, 4}, 2.0f);
    Concat layer({a.get(), b.get()}, ConcatAxis::Cols);

    layer.forward();
    cudaDeviceSynchronize();

    auto host = layer.data().to_host();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 2; ++c)
            EXPECT_NEAR(host(r, c), 1.0f, 1e-5f);
        for (int c = 2; c < 6; ++c)
            EXPECT_NEAR(host(r, c), 2.0f, 1e-5f);
    }
}

TEST(Layer, Select_Forward) {
    HostMatrix<float> src({4, 2});
    src(0, 0) = 1.0f;
    src(1, 0) = 2.0f;
    src(2, 0) = 3.0f;
    src(3, 0) = 4.0f;
    src(0, 1) = 5.0f;
    src(1, 1) = 6.0f;
    src(2, 1) = 7.0f;
    src(3, 1) = 8.0f;

    Param inp({4, 2});
    inp.data().upload(src);
    inp.grad().clear();

    BucketIndex bidx(2, 2);
    HostMatrix<int> idx_src({1, 2});
    idx_src(0, 0) = 0;
    idx_src(0, 1) = 1;
    bidx.data().upload(idx_src);

    Select sel(&inp, &bidx);
    sel.forward();
    cudaDeviceSynchronize();

    auto host = sel.data().to_host();
    EXPECT_NEAR(host(0, 0), 1.0f, 1e-5f);
    EXPECT_NEAR(host(1, 0), 2.0f, 1e-5f);
    EXPECT_NEAR(host(0, 1), 7.0f, 1e-5f);
    EXPECT_NEAR(host(1, 1), 8.0f, 1e-5f);
}

TEST(Layer, SparseAffine_Forward) {
    HostMatrix<float> w_src({2, 2});
    w_src(0, 0) = 1.0f;
    w_src(1, 0) = 2.0f;
    w_src(0, 1) = 3.0f;
    w_src(1, 1) = 4.0f;
    Param weight({2, 2});
    weight.data().upload(w_src);
    weight.grad().clear();

    HostMatrix<float> b_src({2, 1});
    b_src(0, 0) = 0.5f;
    b_src(1, 0) = 0.5f;
    Param bias({2, 1});
    bias.data().upload(b_src);
    bias.grad().clear();

    Input<int> indices({2, 2});
    HostMatrix<int> idx_src({2, 2});
    idx_src(0, 0) = 0;
    idx_src(1, 0) = -1;
    idx_src(0, 1) = 1;
    idx_src(1, 1) = -1;
    indices.data().upload(idx_src);

    SparseAffine layer(&indices, &weight, &bias);
    layer.forward();
    cudaDeviceSynchronize();

    auto host = layer.data().to_host();
    EXPECT_NEAR(host(0, 0), 1.5f, 1e-4f);
    EXPECT_NEAR(host(1, 0), 2.5f, 1e-4f);
    EXPECT_NEAR(host(0, 1), 3.5f, 1e-4f);
    EXPECT_NEAR(host(1, 1), 4.5f, 1e-4f);
}

TEST(Layer, SoftmaxCrossEntropy_Forward_Outputs_Positive) {
    const int C = 4, B = 3;
    auto logits = test::make_param({C, B}, -1.0f, 1.0f);
    auto labels = test::make_input_int(B, {0, 2, 1});

    SoftmaxCrossEntropy layer(logits.get(), labels.get());
    layer.forward();
    cudaDeviceSynchronize();

    auto host = layer.data().to_host();
    for (int c = 0; c < B; ++c)
        EXPECT_GT((double)host(0, c), 0.0);
}

TEST(Layer, SoftmaxCrossEntropy_Forward_PerfectPrediction_SmallLoss) {
    const int C = 3, B = 1;
    HostMatrix<float> src({C, B});
    src(0, 0) = 100.0f;
    src(1, 0) = 0.0f;
    src(2, 0) = 0.0f;

    Param logits({C, B});
    logits.data().upload(src);
    logits.grad().clear();

    auto labels = test::make_input_int(B, {0});
    SoftmaxCrossEntropy layer(&logits, labels.get());
    layer.forward();
    cudaDeviceSynchronize();

    auto loss = layer.data().to_host()(0, 0);
    EXPECT_LT((double)loss, 0.01);
}

TEST(Graph, Builder_AffineLayer) {
    Graph g;
    GraphBuilder b(g);
    auto x = b.input_float("x", {6, 8});
    auto fc = b.affine_layer(6, 4);
    auto y = fc(x);
    EXPECT_EQ(y.get()->shape().rows(), 4);
    EXPECT_EQ(y.get()->shape().cols(), 8);
}

TEST(Graph, Builder_Concat) {
    Graph g;
    GraphBuilder b(g);
    auto x = b.input_float("x", {4, 3});
    auto y = b.input_float("y", {5, 3});
    auto c = b.concat({x, y}, ConcatAxis::Rows);
    EXPECT_EQ(c.get()->shape().rows(), 9);
    EXPECT_EQ(c.get()->shape().cols(), 3);
}
