#pragma once

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "layer/include.h"

namespace nn::network {

class Network {
  public:
    Network(
        std::vector<layer::Layer*> layers, layer::Layer* prediction, layer::Layer* loss = nullptr
    )
        : layers_(layers),
          prediction_(dynamic_cast<layer::TypedLayer<float>*>(prediction)),
          loss_(dynamic_cast<layer::TypedLayer<float>*>(loss)) {

        for (auto* layer : layers_)
            CHECK(layer);

        CHECK(prediction_);
        CHECK(contains(layers_, prediction_));
        CHECK(!loss_ || contains(layers_, loss_));

        if (loss_)
            CHECK(loss_->shape().rows() == 1);

        kernel::cublas::create();
    }

    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    Network(Network&&) = delete;
    Network& operator=(Network&&) = delete;

    ~Network() { kernel::cublas::destroy(); }

    void forward() {
        for (auto* layer : layers_)
            layer->forward();

        if (loss_) {
            running_loss_.resize(loss_->shape());
            kernel::elemwise_binary_forward(
                running_loss_, loss_->data(), running_loss_, kernel::AddBinary{}
            );
        }
    }

    void backward() {
        CHECK(loss_);

        zero_grads();

        kernel::set(loss_->grad(), 1.0f);
        for (int i = (int)layers_.size() - 1; i >= 0; --i)
            layers_[i]->backward();
    }

    const std::vector<layer::Param*> params() const {
        std::vector<layer::Param*> result;

        std::unordered_set<layer::Param*> seen;
        for (auto* layer : layers_)
            if (auto* t = dynamic_cast<layer::Param*>(layer))
                if (seen.insert(t).second)
                    result.push_back(t);

        return result;
    }

    data::GPUMatrix<float>& prediction() { return prediction_->data(); }
    data::GPUMatrix<float>& running_loss() { return running_loss_; }

  private:
    std::vector<layer::Layer*> layers_;
    data::GPUMatrix<float> running_loss_;
    layer::TypedLayer<float>* prediction_;
    layer::TypedLayer<float>* loss_;

    static bool contains(const std::vector<layer::Layer*>& layers, layer::Layer* target) {
        return std::find(layers.begin(), layers.end(), target) != layers.end();
    }

    void zero_grads() {
        std::unordered_set<data::GPUMatrix<float>*> seen;
        for (auto* layer : layers_) {
            if (layer == loss_)
                continue;

            auto* layer_t = dynamic_cast<layer::TypedLayer<float>*>(layer);
            if (!layer_t)
                continue;

            auto* g = &layer_t->grad();
            if (g && !g->empty() && seen.insert(g).second)
                g->clear();
        }
    }
};

} // namespace nn::network
