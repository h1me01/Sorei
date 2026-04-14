#pragma once

#include <cuda_runtime.h>

#include "sorei/nn.h"

#include "framework.h"
#include "grad_check.h"

using namespace sorei;
using namespace sorei::tensor;
using namespace sorei::nn;
using namespace sorei::nn::layer;
using namespace sorei::cuda;

// Layer shape inference

TEST(Layer, Param_Shape) {
    Param p(Shape(8, 4));
    EXPECT_EQ(p.data().rows(), 8);
    EXPECT_EQ(p.data().cols(), 4);
}

TEST(Layer, InputFloat_Shape) {
    InputFloat inp(Shape(5, 3));
    EXPECT_EQ(inp.data().rows(), 5);
    EXPECT_EQ(inp.data().cols(), 3);
}

TEST(Layer, ElemwiseUnary_Shape) {
    Param p(Shape(6, 4));
    ElemwiseUnary layer(&p, ReLU{});
    EXPECT_EQ(layer.data().rows(), 6);
    EXPECT_EQ(layer.data().cols(), 4);
}

TEST(Layer, ElemwiseBinary_Shape_NonBroadcast) {
    Param a(Shape(4, 3));
    Param b(Shape(4, 3));
    ElemwiseBinary layer(&a, &b, AddBinary{});
    EXPECT_EQ(layer.data().rows(), 4);
    EXPECT_EQ(layer.data().cols(), 3);
}

TEST(Layer, ElemwiseBinary_Shape_Broadcast) {
    Param bias(Shape(4, 1));
    Param data(Shape(4, 5));
    ElemwiseBinary layer(&bias, &data, AddBinary{});
    EXPECT_EQ(layer.data().rows(), 4);
    EXPECT_EQ(layer.data().cols(), 5);
}

TEST(Layer, MatMul_Shape) {
    Param w(Shape(8, 6));
    Param x(Shape(6, 4));
    MatMul layer(&w, &x);
    EXPECT_EQ(layer.data().rows(), 8);
    EXPECT_EQ(layer.data().cols(), 4);
}

TEST(Layer, Affine_Shape) {
    Param w(Shape(8, 6));
    Param x(Shape(6, 4));
    Param b(Shape(8, 1));
    Affine layer(&x, &w, &b);
    EXPECT_EQ(layer.data().rows(), 8);
    EXPECT_EQ(layer.data().cols(), 4);
}

TEST(Layer, Mean_Shape) {
    Param p(Shape(5, 8));
    Mean layer(&p);
    EXPECT_EQ(layer.data().rows(), 1);
    EXPECT_EQ(layer.data().cols(), 1);
}

TEST(Layer, PairwiseMul_Shape) {
    Param p(Shape(8, 3));
    PairwiseMul layer(&p);
    EXPECT_EQ(layer.data().rows(), 4);
    EXPECT_EQ(layer.data().cols(), 3);
}

TEST(Layer, Concat_Rows_Shape) {
    Param a(Shape(3, 4));
    Param b(Shape(5, 4));
    Concat layer({&a, &b}, ConcatAxis::Rows);
    EXPECT_EQ(layer.data().rows(), 8);
    EXPECT_EQ(layer.data().cols(), 4);
}

TEST(Layer, Concat_Cols_Shape) {
    Param a(Shape(4, 2));
    Param b(Shape(4, 3));
    Concat layer({&a, &b}, ConcatAxis::Cols);
    EXPECT_EQ(layer.data().rows(), 4);
    EXPECT_EQ(layer.data().cols(), 5);
}

TEST(Layer, SoftmaxCrossEntropy_Shape) {
    Param logits(Shape(10, 5));
    Input<int> labels(Shape(1, 5));
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
    Param inp(Shape(count * out_dim, batch));
    BucketIndex bidx(count, batch);
    Select sel(&inp, &bidx);
    EXPECT_EQ(sel.data().rows(), out_dim);
    EXPECT_EQ(sel.data().cols(), batch);
}

// Layer forward correctness

TEST(Layer, ElemwiseUnary_Forward_ReLU_Correct) {
    auto p = test::make_param_filled(4, 1, -1.0f);
    ElemwiseUnary layer(p.get(), ReLU{});

    layer.forward();
    cudaDeviceSynchronize();

    auto host = layer.data().to_host();
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(host(i), 0.0f, 1e-6f);
}

TEST(Layer, ElemwiseBinary_Forward_Add_Correct) {
    auto a = test::make_param_filled(3, 2, 1.0f);
    auto b = test::make_param_filled(3, 2, 2.0f);
    ElemwiseBinary layer(a.get(), b.get(), AddBinary{});

    layer.forward();
    cudaDeviceSynchronize();

    auto host = layer.data().to_host();
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(host(i), 3.0f, 1e-5f);
}

TEST(Layer, MatMul_Forward_Correct) {
    auto w = test::make_param_filled(2, 3, 1.0f);
    auto x = test::make_param_filled(3, 4, 1.0f);

    MatMul layer(w.get(), x.get());
    layer.forward();
    cudaDeviceSynchronize();

    auto host = layer.data().to_host();
    for (int i = 0; i < 8; ++i)
        EXPECT_NEAR(host(i), 3.0f, 1e-4f);
}

TEST(Layer, Mean_Forward_Correct) {
    HostMatrix<float> src(Shape(3, 4));
    for (int i = 0; i < 12; ++i)
        src(i) = (float)i;

    Param p(Shape(3, 4));
    p.data().upload(src);
    p.grad().clear();

    Mean layer(&p);
    layer.forward();
    cudaDeviceSynchronize();
    EXPECT_NEAR(layer.data().to_host()(0), 5.5f, 1e-4f);
}

TEST(Layer, PairwiseMul_Forward_Correct) {
    HostMatrix<float> src(Shape(4, 1));
    src(0, 0) = 1;
    src(1, 0) = 2;
    src(2, 0) = 3;
    src(3, 0) = 4;

    Param p(Shape(4, 1));
    p.data().upload(src);
    p.grad().clear();

    PairwiseMul layer(&p);
    layer.forward();
    cudaDeviceSynchronize();

    auto host = layer.data().to_host();
    EXPECT_NEAR(host(0, 0), 3.0f, 1e-5f);
    EXPECT_NEAR(host(1, 0), 8.0f, 1e-5f);
}

TEST(Layer, SoftmaxCrossEntropy_Forward_Outputs_Positive) {
    const int C = 4, B = 3;
    auto logits = test::make_param(C, B, -1.0f, 1.0f);
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
    HostMatrix<float> src(Shape(C, B));
    src(0, 0) = 100.0f;
    src(1, 0) = 0.0f;
    src(2, 0) = 0.0f;

    Param logits(Shape(C, B));
    logits.data().upload(src);
    logits.grad().clear();

    auto labels = test::make_input_int(B, {0});
    SoftmaxCrossEntropy layer(&logits, labels.get());
    layer.forward();
    cudaDeviceSynchronize();

    auto loss = layer.data().to_host()(0, 0);
    EXPECT_LT((double)loss, 0.01);
}

// GraphBuilder fluent API

TEST(Graph, Builder_BasicInputs) {
    graph::Graph g;
    graph::GraphBuilder b(g);
    auto x = b.input_float(Shape(8, 4), "x");
    auto lbl = b.input_int(Shape(1, 4), "labels");
    EXPECT_TRUE((bool)x);
    EXPECT_TRUE((bool)lbl);
    EXPECT_EQ(x.get()->shape().rows(), 8);
}

TEST(Graph, Builder_AffineLayer_Shape) {
    graph::Graph g;
    graph::GraphBuilder b(g);
    auto x = b.input_float(Shape(6, 8), "x");
    auto fc = b.affine_layer(6, 4, "fc1");
    auto y = fc(x);
    EXPECT_EQ(y.get()->shape().rows(), 4);
    EXPECT_EQ(y.get()->shape().cols(), 8);
}

TEST(Graph, Builder_FluentActivations) {
    graph::Graph g;
    graph::GraphBuilder b(g);
    auto x = b.input_float(Shape(4, 2), "x");
    auto r = x.relu();
    auto s = x.sigmoid();
    auto c = x.clamped_relu();
    EXPECT_TRUE((bool)r);
    EXPECT_TRUE((bool)s);
    EXPECT_TRUE((bool)c);
}

TEST(Graph, Builder_FluentArithmetic) {
    graph::Graph g;
    graph::GraphBuilder b(g);
    auto x = b.input_float(Shape(4, 2), "x");
    auto y = b.input_float(Shape(4, 2), "y");
    EXPECT_TRUE((bool)(x + y));
    EXPECT_TRUE((bool)(x - y));
    EXPECT_TRUE((bool)(x * y));
    EXPECT_TRUE((bool)(x + 1.0f));
    EXPECT_TRUE((bool)(x * 2.0f));
    EXPECT_TRUE((bool)(2.0f * x));
}

TEST(Graph, Builder_Concat) {
    graph::Graph g;
    graph::GraphBuilder b(g);
    auto x = b.input_float(Shape(4, 3), "x");
    auto y = b.input_float(Shape(5, 3), "y");
    auto c = b.concat({x, y}, layer::ConcatAxis::Rows);
    EXPECT_EQ(c.get()->shape().rows(), 9);
    EXPECT_EQ(c.get()->shape().cols(), 3);
}

TEST(Graph, Builder_SoftmaxCrossEntropy) {
    graph::Graph g;
    graph::GraphBuilder b(g);
    auto logits = b.input_float(Shape(10, 4), "logits");
    auto labels = b.input_int(Shape(1, 4), "labels");
    auto loss = logits.softmax_cross_entropy(labels);
    auto mean = loss.mean();
    EXPECT_EQ(mean.get()->shape().rows(), 1);
    EXPECT_EQ(mean.get()->shape().cols(), 1);
}

TEST(Graph, TopologicalSort_Correctness) {
    graph::Graph g;
    graph::GraphBuilder b(g);

    auto x = b.input_float(Shape(4, 2), "x");
    auto r = x.relu();
    auto s = r.sigmoid();
    auto sorted = g.topological_sort();

    EXPECT_GE((int)sorted.size(), 3);

    for (int i = 0; i < (int)sorted.size(); ++i) {
        for (auto* inp : sorted[i]->inputs()) {
            bool inp_before = false;
            for (int j = 0; j < i; ++j)
                if (sorted[j] == inp) {
                    inp_before = true;
                    break;
                }
            EXPECT_TRUE(inp_before);
        }
    }
}

TEST(Graph, NamedLookup) {
    graph::Graph g;
    graph::GraphBuilder b(g);
    auto x = b.input_float(Shape(4, 2), "my_input");
    auto* found = g.get<InputFloat>("my_input");
    EXPECT_TRUE(found != nullptr);
    EXPECT_EQ(found->shape().rows(), 4);
}

// Network forward / backward / parameters

TEST(Network, ForwardBackward_NoNaN) {
    graph::Graph g;
    graph::GraphBuilder b(g);

    auto x = b.input_float(Shape(4, 8), "x");
    auto fc1 = b.affine_layer(4, 3, "fc1");
    auto fc2 = b.affine_layer(3, 1, "fc2");
    auto y = fc2(fc1(x).relu());
    auto lbl = b.input_int(Shape(1, 8), "lbl");
    auto loss = y.softmax_cross_entropy(lbl).mean();

    network::Network net(g.topological_sort(), y.get(), loss.get());

    {
        HostMatrix<float> xdata(Shape(4, 8));
        xdata.fill(0.1f);
        auto* xinp = g.get<InputFloat>("x");
        xinp->data().upload(xdata);
        HostMatrix<int> ldata(Shape(1, 8));
        for (int c = 0; c < 8; ++c)
            ldata(0, c) = 0;
        auto* linp = g.get<InputInt>("lbl");
        linp->data().upload(ldata);
    }

    net.forward();
    cudaDeviceSynchronize();
    net.backward();
    cudaDeviceSynchronize();

    auto pred = net.prediction().to_host();
    EXPECT_TRUE(pred.shape() == Shape(1, 8));
    for (int i = 0; i < pred.size(); ++i)
        EXPECT_FALSE(std::isnan(pred(i)));
}

TEST(Network, RunningLoss_Accumulates) {
    graph::Graph g;
    graph::GraphBuilder b(g);
    auto x = b.input_float(Shape(2, 4), "x");
    auto lbl = b.input_int(Shape(1, 4), "lbl");
    auto fc = b.affine_layer(2, 2, "fc");
    auto pred = fc(x);
    auto loss = pred.softmax_cross_entropy(lbl).mean();

    network::Network net(g.topological_sort(), pred.get(), loss.get());

    auto* xinp = g.get<InputFloat>("x");
    auto* linp = g.get<InputInt>("lbl");
    HostMatrix<float> xdata(Shape(2, 4));
    xdata.fill(0.1f);
    HostMatrix<int> ldata(Shape(1, 4));
    for (int c = 0; c < 4; ++c)
        ldata(0, c) = 0;
    xinp->data().upload(xdata);
    linp->data().upload(ldata);

    net.forward();
    cudaDeviceSynchronize();
    auto rl = net.running_loss().to_host();

    EXPECT_EQ(rl.size(), 1);
    EXPECT_GT((double)rl(0), 0.0);
}

TEST(Network, GradsNonZeroAfterBackward) {
    graph::Graph g;
    graph::GraphBuilder b(g);
    auto x = b.input_float(Shape(3, 4), "x");
    auto lbl = b.input_int(Shape(1, 4), "lbl");
    auto fc = b.affine_layer(3, 2, "fc");
    auto out = fc(x);
    auto loss = out.softmax_cross_entropy(lbl).mean();

    network::Network net(g.topological_sort(), out.get(), loss.get());

    HostMatrix<float> xdata(Shape(3, 4));
    xdata.fill(0.5f);
    HostMatrix<int> ldata(Shape(1, 4));
    for (int c = 0; c < 4; ++c)
        ldata(0, c) = c % 2;
    g.get<InputFloat>("x")->data().upload(xdata);
    g.get<InputInt>("lbl")->data().upload(ldata);

    net.forward();
    net.backward();
    cudaDeviceSynchronize();

    for (auto* p : net.params()) {
        auto grad_host = p->grad().to_host();
        float abs_sum = 0.0f;
        for (int i = 0; i < grad_host.size(); ++i)
            abs_sum += std::abs(grad_host(i));
        if (abs_sum > 1e-8f)
            return;
    }
    EXPECT_TRUE(false); // should not reach here
}
