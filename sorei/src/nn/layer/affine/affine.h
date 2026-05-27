#pragma once

#include "../../../cublas/cublas.h"
#include "../layer.h"

namespace sorei::nn {

class Affine : public TypedLayer<float> {
  public:
    Affine(Layer* input, Layer* weight, Layer* bias)
        : TypedLayer<float>("Affine"),
          input_(checked_cast<TypedLayer<float>>(input)),
          weight_(checked_cast<TypedLayer<float>>(weight)),
          bias_(checked_cast<TypedLayer<float>>(bias)) {

        SOREI_CHECK(weight_->shape().cols() == input_->shape().rows());
    }

    void forward() override;
    void backward() override;

    std::vector<LayerInputSlot> mutable_inputs() override {
        return {
            LayerInputSlot::from(input_), LayerInputSlot::from(weight_), LayerInputSlot::from(bias_)
        };
    }

    matrix::Shape shape() const override {
        return {weight_->shape().rows(), input_->shape().cols()};
    }

  private:
    TypedLayer<float>* input_;
    TypedLayer<float>* weight_;
    TypedLayer<float>* bias_;
};

} // namespace sorei::nn
