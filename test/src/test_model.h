#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <filesystem>
#include <string>

#include "sorei/nn.h"

#include "framework.h"

using namespace sorei;
using namespace sorei::tensor;
using namespace sorei::nn;
using namespace sorei::nn::graph;
using namespace sorei::nn::optim;
using namespace sorei::nn::lr_sched;

namespace {

struct TinyMLP : public Model {
    int in_dim, hidden, n_classes;

    TinyMLP(int in_dim, int hidden, int n_classes)
        : in_dim(in_dim),
          hidden(hidden),
          n_classes(n_classes) {}

    GraphOutput build_graph(GraphBuilder& b) override {
        auto x = b.input_float(Shape(in_dim, 0), "x");
        auto lbls = b.input_int(Shape(1, 0), "labels");
        auto fc1 = b.affine_layer(in_dim, hidden, "fc1");
        auto fc2 = b.affine_layer(hidden, n_classes, "fc2");
        auto pred = fc2(fc1(x).relu());
        auto loss = pred.softmax_cross_entropy(lbls).mean();
        return {pred, loss};
    }
};

struct TinyReg : public Model {
    GraphOutput build_graph(GraphBuilder& b) override {
        auto x = b.input_float(Shape(4, 0), "x");
        auto lbls = b.input_int(Shape(1, 0), "labels");
        auto fc = b.affine_layer(4, 2, "fc");
        auto pred = fc(x);
        auto loss = pred.softmax_cross_entropy(lbls).mean();
        return {pred, loss};
    }
};

} // namespace

// Build and forward pass

TEST(Model, Build_ForwardProducesCorrectShape) {
    TinyMLP model(8, 16, 3);

    Tensor<float> x({8, 4});
    x.fill(0.1f);
    Tensor<int> lbl({1, 4});
    for (int i = 0; i < 4; ++i)
        lbl(0, i) = i % 3;

    model.forward({{"x", x}, {"labels", lbl}});
    cudaDeviceSynchronize();

    auto& pred = model.prediction();
    EXPECT_EQ(pred.shape().rows(), 3);
    EXPECT_EQ(pred.shape().cols(), 4);
}

TEST(Model, Forward_OutputNoNaN) {
    TinyMLP model(6, 12, 4);

    Tensor<float> x({6, 8});
    x.fill(0.2f);
    Tensor<int> lbl({1, 8});
    for (int i = 0; i < 8; ++i)
        lbl(0, i) = i % 4;

    model.forward({{"x", x}, {"labels", lbl}});
    cudaDeviceSynchronize();

    auto cpu = model.prediction().to_cpu();
    for (int i = 0; i < cpu.size(); ++i)
        EXPECT_FALSE(std::isnan(cpu(i)));
}

// Running loss

TEST(Model, RunningLoss_Positive) {
    TinyMLP model(4, 8, 2);

    Tensor<float> x({4, 4});
    x.fill(0.5f);
    Tensor<int> lbl({1, 4});
    for (int i = 0; i < 4; ++i)
        lbl(0, i) = i % 2;

    model.forward({{"x", x}, {"labels", lbl}});
    float loss = model.running_loss();
    EXPECT_GT((double)loss, 0.0);
}

TEST(Model, RunningLoss_Accumulates_Over_Steps) {
    TinyMLP model(4, 8, 2);

    Tensor<float> x({4, 4});
    x.fill(0.5f);
    Tensor<int> lbl({1, 4});
    for (int i = 0; i < 4; ++i)
        lbl(0, i) = i % 2;

    model.forward({{"x", x}, {"labels", lbl}});
    float after1 = model.running_loss();
    model.forward({{"x", x}, {"labels", lbl}});
    float after2 = model.running_loss();
    // Accumulated over two steps → should be roughly 2x
    EXPECT_GT((double)after2, (double)after1 * 1.5);
}

TEST(Model, RunningLoss_ClearResetsToZero) {
    TinyMLP model(4, 8, 2);
    Tensor<float> x({4, 4});
    x.fill(0.5f);
    Tensor<int> lbl({1, 4});
    for (int i = 0; i < 4; ++i)
        lbl(0, i) = 0;

    model.forward({{"x", x}, {"labels", lbl}});
    model.clear_running_loss();
    EXPECT_NEAR((double)model.running_loss(), 0.0, 1e-5);
}

// Backward: gradients are non-zero

TEST(Model, Backward_ProducesNonZeroGradients) {
    TinyMLP model(4, 8, 3);
    Tensor<float> x({4, 6});
    x.fill(0.3f);
    Tensor<int> lbl({1, 6});
    for (int i = 0; i < 6; ++i)
        lbl(0, i) = i % 3;

    model.forward({{"x", x}, {"labels", lbl}});
    model.backward();
    cudaDeviceSynchronize();

    bool found_nonzero = false;
    for (auto* p : model.params()) {
        auto g = p->grad().to_cpu();
        for (int i = 0; i < g.size(); ++i) {
            if (std::abs(g(i)) > 1e-9f) {
                found_nonzero = true;
                break;
            }
        }
        if (found_nonzero)
            break;
    }
    EXPECT_TRUE(found_nonzero);
}

// Overfitting a tiny batch

TEST(Model, Overfit_TinyBatch_LossDecreases) {
    TinyMLP model(8, 32, 4);
    AdamW optim(model.params(), 0.9f, 0.999f, 0.0f);

    Tensor<float> x({8, 4});
    x.fill(0.0f);
    Tensor<int> lbl({1, 4});
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 8; ++r)
            x(r, c) = (r == c % 8) ? 1.0f : 0.0f;
        lbl(0, c) = c;
    }

    model.forward({{"x", x}, {"labels", lbl}});
    const float initial_loss = model.running_loss();

    for (int iter = 0; iter < 200; ++iter) {
        model.clear_running_loss();
        model.forward({{"x", x}, {"labels", lbl}});
        model.backward();
        optim.step(1e-2f);
    }

    model.clear_running_loss();
    model.forward({{"x", x}, {"labels", lbl}});
    const float final_loss = model.running_loss();

    EXPECT_LT((double)final_loss, (double)initial_loss * 0.5);
}

// Save / load params produces same output

TEST(Model, LoadedParams_ProduceSameOutput) {
    const std::string path = "/tmp/sorei_test_model_output.bin";

    TinyMLP model_a(4, 8, 2);
    Tensor<float> x({4, 3});
    x.fill(0.7f);
    Tensor<int> lbl({1, 3});
    for (int i = 0; i < 3; ++i)
        lbl(0, i) = i % 2;

    model_a.forward({{"x", x}, {"labels", lbl}});
    auto pred_a = model_a.prediction().to_cpu();
    model_a.save_params(path);

    TinyMLP model_b(4, 8, 2);
    model_b.forward({{"x", x}, {"labels", lbl}});
    model_b.load_params(path);
    model_b.forward({{"x", x}, {"labels", lbl}});
    auto pred_b = model_b.prediction().to_cpu();

    EXPECT_EQ(pred_a.size(), pred_b.size());
    for (int i = 0; i < pred_a.size(); ++i)
        EXPECT_NEAR(pred_a(i), pred_b(i), 1e-5f);

    std::filesystem::remove(path);
}

// get_param by name

TEST(Model, GetParam_ByName) {
    TinyMLP model(4, 8, 2);
    Tensor<float> x({4, 2});
    x.fill(0.1f);
    Tensor<int> lbl({1, 2});
    lbl(0, 0) = 0;
    lbl(0, 1) = 1;
    model.forward({{"x", x}, {"labels", lbl}});

    auto& w = model.get_param("fc1.W");
    EXPECT_EQ(w.shape().rows(), 8);
    EXPECT_EQ(w.shape().cols(), 4);

    auto& b = model.get_param("fc1.B");
    EXPECT_EQ(b.shape().rows(), 8);
}

// Graph optimizer

TEST(Model, GraphOptimizer_FoldSelfMul) {
    struct SquareModel : public Model {
        GraphOutput build_graph(GraphBuilder& b) override {
            auto x = b.input_float(Shape(4, 0), "x");
            auto y = x * x;
            auto loss = y.mean();
            return {y, loss};
        }
    };

    SquareModel model;
    Tensor<float> x({4, 3});
    for (int i = 0; i < 12; ++i)
        x.data()[i] = (float)(i + 1) * 0.1f;

    model.forward({{"x", x}});
    cudaDeviceSynchronize();

    auto pred = model.prediction().to_cpu();
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < 4; ++r) {
            float xi = (float)(4 * c + r + 1) * 0.1f;
            EXPECT_NEAR(pred(r, c), xi * xi, 1e-4f);
        }
    }
}

// Multiple named inputs

TEST(Model, MultipleInputs_BothUsed) {
    struct AddModel : public Model {
        GraphOutput build_graph(GraphBuilder& b) override {
            auto x = b.input_float(Shape(4, 0), "x");
            auto y = b.input_float(Shape(4, 0), "y");
            auto lbl = b.input_int(Shape(1, 0), "lbl");
            auto out = (x + y);
            auto loss = out.softmax_cross_entropy(lbl).mean();
            return {out, loss};
        }
    };

    AddModel model;
    Tensor<float> x({4, 3});
    x.fill(1.0f);
    Tensor<float> y({4, 3});
    y.fill(2.0f);
    Tensor<int> lbl({1, 3});
    for (int i = 0; i < 3; ++i)
        lbl(0, i) = i % 4;

    model.forward({{"x", x}, {"y", y}, {"lbl", lbl}});
    cudaDeviceSynchronize();

    auto pred = model.prediction().to_cpu();
    for (int i = 0; i < pred.size(); ++i)
        EXPECT_NEAR(pred(i), 3.0f, 1e-5f);
}

// Params count matches layer count

TEST(Model, Params_Count) {
    TinyMLP model(4, 8, 3);
    Tensor<float> x({4, 2});
    x.fill(0.1f);
    Tensor<int> lbl({1, 2});
    lbl(0, 0) = 0;
    lbl(0, 1) = 1;
    model.forward({{"x", x}, {"labels", lbl}});
    EXPECT_EQ((int)model.params().size(), 4);
}
