#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <filesystem>
#include <string>

#include "sorei/nn.h"

#include "framework.h"

using namespace sorei;
using namespace sorei::matrix;
using namespace sorei::nn;

namespace {

struct TinyMLP : public Model {
    int in_dim, hidden, n_classes;
    AffineLayer fc1, fc2;

    TinyMLP(int in_dim, int hidden, int n_classes)
        : in_dim(in_dim),
          hidden(hidden),
          n_classes(n_classes) {}

    GraphOutput build_graph(GraphBuilder& b) override {
        auto x = b.input_float("x", {in_dim, 0});
        auto lbls = b.input_int("labels", {1, 0});
        fc1 = b.affine_layer(in_dim, hidden);
        fc2 = b.affine_layer(hidden, n_classes);
        auto pred = fc2(fc1(x).relu());
        auto loss = pred.softmax_cross_entropy(lbls).mean();
        return {pred, loss};
    }
};

} // namespace

TEST(Model, Build_ForwardProducesCorrectShape) {
    TinyMLP model(8, 16, 3);

    HostPinnedMatrix<float> x({8, 4});
    x.fill(0.1f);
    HostPinnedMatrix<int> lbl({1, 4});
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

    HostPinnedMatrix<float> x({6, 8});
    x.fill(0.2f);
    HostPinnedMatrix<int> lbl({1, 8});
    for (int i = 0; i < 8; ++i)
        lbl(0, i) = i % 4;

    model.forward({{"x", x}, {"labels", lbl}});
    cudaDeviceSynchronize();

    auto host = model.prediction().to_host();
    for (int i = 0; i < host.size(); ++i)
        EXPECT_FALSE(std::isnan(host(i)));
}

TEST(Model, RunningLoss_Positive) {
    TinyMLP model(4, 8, 2);

    HostPinnedMatrix<float> x({4, 4});
    x.fill(0.5f);
    HostPinnedMatrix<int> lbl({1, 4});
    for (int i = 0; i < 4; ++i)
        lbl(0, i) = i % 2;

    model.forward({{"x", x}, {"labels", lbl}});
    float loss = model.running_loss();
    EXPECT_GT((double)loss, 0.0);
}

TEST(Model, RunningLoss) {
    TinyMLP model(4, 8, 2);

    HostPinnedMatrix<float> x({4, 4});
    x.fill(0.5f);
    HostPinnedMatrix<int> lbl({1, 4});
    for (int i = 0; i < 4; ++i)
        lbl(0, i) = i % 2;

    model.forward({{"x", x}, {"labels", lbl}});
    float after1 = model.running_loss();
    model.forward({{"x", x}, {"labels", lbl}});
    float after2 = model.running_loss();

    EXPECT_GT((double)after2, (double)after1 * 1.5);

    model.zero_running_loss();
    EXPECT_NEAR((double)model.running_loss(), 0.0, 1e-5);
}

TEST(Model, Backward_ProducesNonZeroGradients) {
    TinyMLP model(4, 8, 3);
    HostPinnedMatrix<float> x({4, 6});
    x.fill(0.3f);
    HostPinnedMatrix<int> lbl({1, 6});
    for (int i = 0; i < 6; ++i)
        lbl(0, i) = i % 3;

    model.forward({{"x", x}, {"labels", lbl}});
    model.backward();
    cudaDeviceSynchronize();

    bool found_nonzero = false;
    for (auto* p : model.params()) {
        auto g = p->grad().to_host();
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

TEST(Model, Overfit_TinyBatch) {
    TinyMLP model(8, 32, 4);
    AdamW optim(model.params(), 0.9f, 0.999f, 0.0f);

    HostPinnedMatrix<float> x({8, 4});
    x.fill(0.0f);
    HostPinnedMatrix<int> lbl({1, 4});
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 8; ++r)
            x(r, c) = (r == c % 8) ? 1.0f : 0.0f;
        lbl(0, c) = c;
    }

    model.forward({{"x", x}, {"labels", lbl}});
    const float initial_loss = model.running_loss();

    for (int iter = 0; iter < 200; ++iter) {
        model.zero_running_loss();
        model.forward({{"x", x}, {"labels", lbl}});
        model.backward();
        optim.step(1e-2f);
    }

    model.zero_running_loss();
    model.forward({{"x", x}, {"labels", lbl}});
    const float final_loss = model.running_loss();

    EXPECT_LT((double)final_loss, (double)initial_loss * 0.5);
}

TEST(Model, LoadedParams_ProduceSameOutput) {
    const std::string path = "/tmp/sorei_test_model_output.bin";

    TinyMLP model_a(4, 8, 2);
    HostPinnedMatrix<float> x({4, 3});
    x.fill(0.7f);
    HostPinnedMatrix<int> lbl({1, 3});
    for (int i = 0; i < 3; ++i)
        lbl(0, i) = i % 2;

    model_a.forward({{"x", x}, {"labels", lbl}});
    auto pred_a = model_a.prediction().to_host();
    model_a.save_params(path);

    TinyMLP model_b(4, 8, 2);
    model_b.forward({{"x", x}, {"labels", lbl}});
    model_b.load_params(path);
    model_b.forward({{"x", x}, {"labels", lbl}});
    auto pred_b = model_b.prediction().to_host();

    EXPECT_EQ(pred_a.size(), pred_b.size());
    for (int i = 0; i < pred_a.size(); ++i)
        EXPECT_NEAR(pred_a(i), pred_b(i), 1e-5f);

    std::filesystem::remove(path);
}
