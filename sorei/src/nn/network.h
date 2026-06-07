#pragma once

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "../cuda/ops.h"
#include "layer/include.h"

namespace sorei::nn {

class Network {
  public:
    Network(std::vector<Layer*> layers, Layer* prediction, Layer* loss = nullptr)
        : layers_(layers),
          prediction_(checked_cast<TypedLayer<float>>(prediction)),
          loss_(checked_cast<TypedLayer<float>>(loss)) {

        for (auto* layer : layers_)
            SOREI_CHECK(layer);
        SOREI_CHECK(prediction_);
        SOREI_CHECK(contains(layers_, prediction_));
        SOREI_CHECK(!loss_ || contains(layers_, loss_));

        if (loss_) {
            SOREI_CHECK(loss_->requires_grad());
            SOREI_CHECK(loss_->shape().size() == 1);
            running_loss_.resize(loss_->shape());
            running_loss_.clear();
            cuda::set(loss_->grad(), 1.0f);
        }
    }

    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    Network(Network&&) = delete;
    Network& operator=(Network&&) = delete;

    void forward() {
        for (auto* layer : layers_)
            layer->forward();
        if (loss_)
            cuda::add(running_loss_, loss_->data(), running_loss_);
    }

    void backward() {
        SOREI_CHECK(loss_);

        for (auto* l : layers_)
            l->reset_grad_write();

        for (int i = (int)layers_.size() - 1; i >= 0; --i)
            layers_[i]->backward();
    }

    std::vector<Param*> params() const {
        std::vector<Param*> result;

        std::unordered_set<Param*> seen;
        for (auto* layer : layers_)
            if (auto* t = dynamic_cast<Param*>(layer))
                if (seen.insert(t).second)
                    result.push_back(t);

        return result;
    }

    matrix::DeviceMatrix<float>& prediction() { return prediction_->data(); }
    matrix::DeviceMatrix<float>& running_loss() { return running_loss_; }

  private:
    std::vector<Layer*> layers_;
    matrix::DeviceMatrix<float> running_loss_;
    TypedLayer<float>* prediction_;
    TypedLayer<float>* loss_;

    static bool contains(const std::vector<Layer*>& layers, Layer* target) {
        return std::ranges::find(layers, target) != layers.end();
    }
};

} // namespace sorei::nn
