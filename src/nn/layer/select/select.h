#pragma once

#include "../layer.h"
#include "bucket_index.h"

namespace sorei::nn::layer {

class Select : public TypedLayer<float> {
  public:
    Select(Layer* input, BucketIndex* bucket)
        : TypedLayer<float>("Select"),
          input_(layer_cast<TypedLayer<float>>(input)),
          bucket_(bucket) {

        CHECK(bucket_);
        CHECK(input_->shape().rows() % bucket_->count() == 0);
    }

    void forward() override;
    void backward() override;

    std::vector<LayerInputSlot> mutable_inputs() override {
        return {LayerInputSlot::from(input_), LayerInputSlot::from(bucket_)};
    }

    tensor::Shape shape() const override {
        return {input_->shape().rows() / bucket_->count(), input_->shape().cols()};
    }

  private:
    TypedLayer<float>* input_;
    BucketIndex* bucket_;
};

} // namespace sorei::nn::layer
