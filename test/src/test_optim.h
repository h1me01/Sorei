#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <filesystem>
#include <string>

#include "sorei/nn.h"

#include "framework.h"
#include "grad_check.h" // make_param

using namespace sorei;
using namespace sorei::cuda;
using namespace sorei::tensor;
using namespace sorei::nn;
using namespace sorei::nn::layer;
using namespace sorei::nn::optim;

namespace {

struct SimpleSquareLoss {
    std::unique_ptr<layer::Param> param;
    std::unique_ptr<ElemwiseUnary> sq;
    std::unique_ptr<Mean> loss;

    explicit SimpleSquareLoss(const Shape& shape, float init_val = 0.5f) {
        param = std::make_unique<layer::Param>(shape);
        HostMatrix<float> m(shape);
        m.fill(init_val);
        param->data().upload(m);
        param->grad().clear();

        sq = std::make_unique<ElemwiseUnary>(param.get(), PowInt{2});
        loss = std::make_unique<Mean>(sq.get());
    }

    void forward_backward() {
        sq->forward();
        loss->forward();
        cudaDeviceSynchronize();

        set(loss->grad(), 1.0f);
        sq->grad().clear();
        param->grad().clear();
        loss->backward();
        sq->backward();
        cudaDeviceSynchronize();
    }

    float loss_value() { return loss->data().to_host()(0); }
};

} // namespace

// Adam

TEST(Optim, Adam_LossDecreases) {
    SimpleSquareLoss net({6, 6}, 1.0f);
    Adam optim({net.param.get()});

    float initial_loss = -1.0f;
    for (int i = 0; i < 50; ++i) {
        net.forward_backward();
        if (i == 0)
            initial_loss = net.loss_value();
        optim.step(1e-2f);
        cudaDeviceSynchronize();
    }

    net.forward_backward();
    float final_loss = net.loss_value();
    EXPECT_LT((double)final_loss, (double)initial_loss * 0.5);
}

TEST(Optim, Adam_ZeroLearningRate_ParamsUnchanged) {
    SimpleSquareLoss net({4, 4}, 0.5f);
    Adam optim({net.param.get()});

    net.forward_backward();
    auto before = net.param->data().to_host();
    optim.step(0.0f);
    cudaDeviceSynchronize();
    auto after = net.param->data().to_host();

    for (int i = 0; i < 16; ++i)
        EXPECT_NEAR(after(i), before(i), 1e-6f);
}

// AdamW

TEST(Optim, AdamW_WeightDecay_MoreAggressive_Than_Adam) {
    SimpleSquareLoss net_adam({4, 4}, 1.0f);
    SimpleSquareLoss net_adamw({4, 4}, 1.0f);

    Adam optim_adam({net_adam.param.get()});
    AdamW optim_adamw({net_adamw.param.get()}, 0.9f, 0.999f, 0.1f);

    for (int i = 0; i < 10; ++i) {
        net_adam.forward_backward();
        net_adamw.forward_backward();
        optim_adam.step(1e-2f);
        optim_adamw.step(1e-2f);
        cudaDeviceSynchronize();
    }

    auto host_adam = net_adam.param->data().to_host();
    auto host_adamw = net_adamw.param->data().to_host();

    float norm_adam = 0.0f, norm_adamw = 0.0f;
    for (int i = 0; i < 16; ++i) {
        norm_adam += host_adam(i) * host_adam(i);
        norm_adamw += host_adamw(i) * host_adamw(i);
    }
    EXPECT_LT((double)norm_adamw, (double)norm_adam);
}

TEST(Optim, AdamW_ParamBounds_Respected) {
    auto p = test::make_param({8, 8}, -2.0f, 2.0f);
    p->set_bounds(-0.5f, 0.5f);

    HostMatrix<float> grad_vals({8, 8});
    grad_vals.fill(1.0f);
    p->grad().upload(grad_vals);

    AdamW optim({p.get()}, 0.9f, 0.999f, 0.0f);
    for (int iter = 0; iter < 100; ++iter) {
        p->grad().upload(grad_vals);
        optim.step(1.0f);
        cudaDeviceSynchronize();
    }

    auto host = p->data().to_host();
    for (int i = 0; i < 64; ++i) {
        EXPECT_GE((double)host(i), -0.5 - 1e-4);
        EXPECT_LE((double)host(i), 0.5 + 1e-4);
    }
}

TEST(Optim, AdamW_SaveLoadState) {
    const std::string state_dir = "/tmp/sorei_test_adamw_state";

    auto p = test::make_param({4, 4});
    AdamW optim({p.get()}, 0.9f, 0.999f, 0.0f);

    HostMatrix<float> g({4, 4});
    g.fill(0.1f);
    for (int i = 0; i < 5; ++i) {
        p->grad().upload(g);
        optim.step(1e-3f);
        cudaDeviceSynchronize();
    }

    optim.save_state(state_dir);

    auto p2 = test::make_param({4, 4});
    p2->data().upload(p->data().to_host());
    p2->grad().upload(p->grad().to_host());

    AdamW optim2({p2.get()}, 0.9f, 0.999f, 0.0f);
    optim2.load_state(state_dir);

    for (int i = 0; i < 5; ++i) {
        optim.step(1e-3f);
        optim2.step(1e-3f);
        cudaDeviceSynchronize();
    }

    auto host1 = p->data().to_host();
    auto host2 = p2->data().to_host();
    for (int i = 0; i < 16; ++i)
        EXPECT_NEAR(host1(i), host2(i), 1e-6f);

    std::filesystem::remove_all(state_dir);
}

TEST(Optim, Adam_MultipleParms_AllUpdated) {
    auto p1 = test::make_param({4, 4}, 1.0f, 1.0f);
    auto p2 = test::make_param({3, 3}, 1.0f, 1.0f);

    HostMatrix<float> g1({4, 4});
    g1.fill(0.5f);
    HostMatrix<float> g2({3, 3});
    g2.fill(0.5f);
    p1->grad().upload(g1);
    p2->grad().upload(g2);

    Adam optim({p1.get(), p2.get()});
    optim.step(1e-1f);
    cudaDeviceSynchronize();

    auto host1 = p1->data().to_host();
    auto host2 = p2->data().to_host();

    for (int i = 0; i < 16; ++i)
        EXPECT_LT((double)host1(i), 1.0);
    for (int i = 0; i < 9; ++i)
        EXPECT_LT((double)host2(i), 1.0);
}
