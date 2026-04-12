#pragma once

#include "sorei/nn.h"

struct MNISTModel : public sorei::nn::Model {
    static constexpr int INPUT_DIM = 28 * 28;
    static constexpr int L1_SIZE = 256;
    static constexpr int L2_SIZE = 128;
    static constexpr int NUM_CLASSES = 10;

    void feed(const sorei::nn::Tensor<float>& images, const sorei::nn::Tensor<int>& labels) {
        forward({{"images", images}, {"labels", labels}});
    }

    sorei::nn::GraphOutput build_graph(sorei::nn::graph::GraphBuilder& b) override {
        auto images = b.input_float({INPUT_DIM, 0}, "images");
        auto labels = b.input_int({1, 0}, "labels");

        // layers
        auto l1 = b.affine_layer(INPUT_DIM, L1_SIZE);
        auto l2 = b.affine_layer(L1_SIZE, L2_SIZE);
        auto l3 = b.affine_layer(L2_SIZE, NUM_CLASSES);

        // forward pass
        auto l1_out = l1(images).relu();
        auto l2_out = l2(l1_out).relu();
        auto logits = l3(l2_out);

        // loss
        auto loss = logits.softmax_cross_entropy(labels).mean();

        return {.prediction = logits, .loss = loss};
    }
};
