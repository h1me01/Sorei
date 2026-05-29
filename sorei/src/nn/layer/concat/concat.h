#pragma once

#include "../layer.h"

namespace sorei::nn {

enum class ConcatAxis { Rows, Cols };

class ConcatBase : public TypedLayer<float> {
  public:
    ConcatBase(std::vector<Layer*> inputs, ConcatAxis axis)
        : TypedLayer<float>("Concat"),
          axis_(axis) {

        SOREI_CHECK(!inputs.empty());
        for (auto* p : inputs)
            inputs_.push_back(checked_cast<TypedLayer<float>>(p));
    }

    ConcatAxis axis() const { return axis_; }

    std::vector<LayerInputSlot> mutable_inputs() override {
        std::vector<LayerInputSlot> ptrs;
        for (auto& p : inputs_)
            ptrs.push_back(LayerInputSlot::from(p));
        return ptrs;
    }

    matrix::Shape shape() const override {
        bool is_vertical = (axis_ == ConcatAxis::Rows);
        const int fixed = is_vertical ? inputs_[0]->shape().cols() : inputs_[0]->shape().rows();

        int total = 0;
        for (const auto& input : inputs_) {
            auto s = input->shape();
            SOREI_CHECK((is_vertical ? s.cols() : s.rows()) == fixed);
            total += is_vertical ? s.rows() : s.cols();
        }

        return is_vertical ? matrix::Shape{total, fixed} : matrix::Shape{fixed, total};
    }

  protected:
    ConcatAxis axis_;
    std::vector<TypedLayer<float>*> inputs_;
};

struct Concat : ConcatBase {
    Concat(std::vector<Layer*> inputs, ConcatAxis axis)
        : ConcatBase(std::move(inputs), axis) {}

    void forward() override;
    void backward() override;
};

struct FusedConcat : ConcatBase {
    FusedConcat(std::vector<Layer*> inputs)
        : ConcatBase(std::move(inputs), ConcatAxis::Rows) {}

    void forward() override {}
    void backward() override {}

    int offset_of(const Layer* layer) const {
        int offset = 0;
        for (const auto& input : inputs_) {
            if (input == layer)
                return offset;
            offset += input->shape().rows();
        }
        SOREI_CHECK(false);
        return -1;
    }
};

} // namespace sorei::nn
