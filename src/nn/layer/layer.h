#pragma once

#include <string>
#include <type_traits>
#include <vector>

#include "../../kernel/include.h"
#include "../../tensor/include.h"

namespace sorei::nn::layer {

class Layer;

class LayerInputSlot {
  public:
    template <typename T>
    static LayerInputSlot from(T*& slot) {
        static_assert(std::is_base_of_v<Layer, T>);

        return LayerInputSlot(
            &slot,
            [](void* ptr) -> Layer* { return *static_cast<T**>(ptr); },
            [](void* ptr, Layer* layer) {
                auto* typed = dynamic_cast<T*>(layer);
                CHECK(typed);
                *static_cast<T**>(ptr) = typed;
            }
        );
    }

    Layer* get() const { return getter_(slot_); }
    void set(Layer* layer) const { setter_(slot_, layer); }

  private:
    using Getter = Layer* (*)(void*);
    using Setter = void (*)(void*, Layer*);

    LayerInputSlot(void* slot, Getter getter, Setter setter)
        : slot_(slot),
          getter_(getter),
          setter_(setter) {}

    void* slot_;
    Getter getter_;
    Setter setter_;
};

class Layer {
  public:
    Layer(std::string name)
        : name_(std::move(name)) {}

    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    Layer(Layer&&) = delete;
    Layer& operator=(Layer&&) = delete;

    virtual ~Layer() = default;

    virtual void forward() {}
    virtual void backward() {}

    virtual std::vector<LayerInputSlot> mutable_inputs() { return {}; }

    std::vector<Layer*> inputs() const {
        std::vector<Layer*> result;
        for (const auto& input : const_cast<Layer*>(this)->mutable_inputs())
            result.push_back(input.get());
        return result;
    }

    void replace_input(Layer* from, Layer* to) {
        for (const auto& input : mutable_inputs())
            if (input.get() == from)
                input.set(to);
    }

    virtual bool requires_grad() const {
        for (auto* input : inputs())
            if (input->requires_grad())
                return true;
        return false;
    }

    virtual tensor::Shape shape() const = 0;
    std::string name() const { return name_; }

  private:
    std::string name_;
};

template <typename T>
class TypedLayer : public Layer {
  public:
    TypedLayer(std::string name)
        : Layer(std::move(name)) {}

    TypedLayer(const TypedLayer&) = delete;
    TypedLayer& operator=(const TypedLayer&) = delete;
    TypedLayer(TypedLayer&&) = delete;
    TypedLayer& operator=(TypedLayer&&) = delete;

    virtual ~TypedLayer() = default;

    tensor::GPUMatrix<T>& data() {
        if (!drop_buffers_)
            data_.resize(shape());
        return data_;
    }

    tensor::GPUMatrix<T>& grad() {
        if (requires_grad() && !drop_buffers_)
            grad_.resize(shape());
        return grad_;
    }

  protected:
    void drop_buffers() {
        data_ = tensor::GPUMatrix<T>();
        grad_ = tensor::GPUMatrix<T>();
        drop_buffers_ = true;
    }

  private:
    bool drop_buffers_ = false;
    tensor::GPUMatrix<T> data_;
    tensor::GPUMatrix<T> grad_;
};

template <typename T>
T* layer_cast(Layer* layer) {
    CHECK(layer);
    T* typed = dynamic_cast<T*>(layer);
    CHECK(typed);
    return typed;
}

} // namespace sorei::nn::layer
