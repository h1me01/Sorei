#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "graph.h"

namespace nn::graph {

class GraphBuilder;

// Node

class Node {
  public:
    Node() = default;

    Node(GraphBuilder* gb, layer::Layer* op)
        : gb_(gb),
          layer_(op) {
        CHECK(layer_);
    }

    Node relu() const;
    // relu clamped to [0, 1]
    Node clamped_relu() const;
    // relu clamped to [0, 1] and squared
    Node squared_clamped_relu() const;
    Node sigmoid() const;
    Node abs() const;
    Node neg() const;

    Node pairwise_mul() const;
    Node mean() const;
    Node pow(float e) const;
    Node clamp(float lo, float hi) const;
    Node select(layer::BucketIndex* index) const;
    Node repeat(int count) const;

    Node operator+(const Node& rhs) const;
    Node operator-(const Node& rhs) const;
    Node operator*(const Node& rhs) const;
    Node operator/(const Node& rhs) const;

    Node operator+(float s) const;
    Node operator-(float s) const;
    Node operator*(float s) const;
    Node operator/(float s) const;

    GraphBuilder& gb() const {
        CHECK(gb_);
        return *gb_;
    }

    virtual layer::Layer* get() const { return layer_; }
    explicit operator bool() const { return layer_ != nullptr; }

  protected:
    GraphBuilder* gb_ = nullptr;
    layer::Layer* layer_ = nullptr;
};

Node operator+(float s, const Node& n);
Node operator-(float s, const Node& n);
Node operator*(float s, const Node& n);
Node operator/(float s, const Node& n);

// Param

struct Param : Node {
    Param() = default;

    Param(GraphBuilder* gb, layer::Layer* op)
        : Node(gb, op) {}

    void uniform_init(float lo, float hi) { get()->uniform_init(lo, hi); }
    void he_init() { get()->he_init(); }
    void set_bounds(float lo, float hi) { get()->set_bounds(lo, hi); }

    data::GPUMatrix<float>& data() { return get()->data(); }
    const data::GPUMatrix<float>& data() const { return get()->data(); }

    int input_dim() const { return get()->data().shape().cols(); }
    int output_dim() const { return get()->data().shape().rows(); }

    layer::Param* get() const { return dynamic_cast<layer::Param*>(Node::get()); }
};

// AffineLayer

struct AffineLayer {
    Param weight;
    Param bias;

    AffineLayer() = default;

    AffineLayer(Param w, Param b)
        : weight(std::move(w)),
          bias(std::move(b)) {}

    Node operator()(const Node& input) const;
};

// GraphBuilder

class GraphBuilder {
  public:
    GraphBuilder(Graph& graph)
        : graph_(graph) {}

    GraphBuilder(const GraphBuilder&) = delete;
    GraphBuilder& operator=(const GraphBuilder&) = delete;
    GraphBuilder(GraphBuilder&&) = delete;
    GraphBuilder& operator=(GraphBuilder&&) = delete;

    Node input_int(const data::Shape& shape, const std::string& name = "") {
        return {this, graph_.emplace_named<layer::InputInt>(name, "InputInt", shape)};
    }

    Node input_float(const data::Shape& shape, const std::string& name = "") {
        return {this, graph_.emplace_named<layer::InputFloat>(name, "InputFloat", shape)};
    }

    layer::BucketIndex* bucket_index(int count, int size, const std::string& name = "") {
        return graph_.emplace_named<layer::BucketIndex>(name, "BucketIndex", count, size);
    }

    Param param(int input_dim, int output_dim, const std::string& name = "") {
        return {
            this,
            graph_.emplace_named<layer::Param>(name, "Param", data::Shape{output_dim, input_dim})
        };
    }

    AffineLayer affine_layer(int input_dim, int output_dim, const std::string& name_prefix = "") {
        const std::string prefix = name_prefix.empty()
                                       ? ("Affine" + std::to_string(affine_param_group_count_++))
                                       : name_prefix;

        auto w = param(input_dim, output_dim, prefix + "_W");
        w.he_init();
        auto b = param(1, output_dim, prefix + "_B");
        return {w, b};
    }

    Node relu(const Node& x) { return {this, make<layer::ElemwiseUnary>(x.get(), kernel::ReLU{})}; }

    // relu clamped to [0, 1]
    Node clamped_relu(const Node& x) {
        return {this, make<layer::ElemwiseUnary>(x.get(), kernel::ClampedReLU{})};
    }

    // relu clamped to [0, 1] and squared
    Node squared_clamped_relu(const Node& x) {
        return {this, make<layer::ElemwiseUnary>(x.get(), kernel::SquaredClampedReLU{})};
    }

    Node sigmoid(const Node& x) {
        return {this, make<layer::ElemwiseUnary>(x.get(), kernel::Sigmoid{})};
    }

    Node abs(const Node& x) { return {this, make<layer::ElemwiseUnary>(x.get(), kernel::Abs{})}; }

    Node neg(const Node& x) { return affine_unary(x, -1.0f, 0.0f); }

    Node clamp(const Node& x, float lo, float hi) {
        return {this, make<layer::ElemwiseUnary>(x.get(), kernel::Clamp{lo, hi})};
    }

    Node pow(const Node& x, float e) {
        if (std::fabs(e) <= 16777216.0f) {
            int n = static_cast<int>(e);
            if (std::fabs(e - n) < 1e-6f)
                return {this, make<layer::ElemwiseUnary>(x.get(), kernel::PowInt{n})};
        }
        return {this, make<layer::ElemwiseUnary>(x.get(), kernel::PowFloat{e})};
    }

    Node add(const Node& x, float s) { return affine_unary(x, 1.0f, s); }
    Node sub(const Node& x, float s) { return affine_unary(x, 1.0f, -s); }
    Node sub(float s, const Node& x) { return affine_unary(x, -1.0f, s); }
    Node mul(const Node& x, float s) { return affine_unary(x, s, 0.0f); }

    Node div(const Node& x, float s) {
        if (s == 0.0f)
            error("GraphBuilder: division by zero");
        return affine_unary(x, 1.0f / s, 0.0f);
    }

    Node div(float s, const Node& x) {
        return {this, make<layer::ElemwiseUnary>(x.get(), kernel::DivLeftUnary{s})};
    }

    Node add(const Node& a, const Node& b) {
        return {this, make<layer::ElemwiseBinary>(a.get(), b.get(), kernel::AddBinary{})};
    }

    Node sub(const Node& a, const Node& b) {
        return {this, make<layer::ElemwiseBinary>(a.get(), b.get(), kernel::SubBinary{})};
    }

    Node mul(const Node& a, const Node& b) {
        return {this, make<layer::ElemwiseBinary>(a.get(), b.get(), kernel::MulBinary{})};
    }

    Node div(const Node& a, const Node& b) {
        return {this, make<layer::ElemwiseBinary>(a.get(), b.get(), kernel::DivBinary{})};
    }

    Node mat_mul(const Node& weight, const Node& input) {
        return {this, make<layer::MatMul>(weight.get(), input.get())};
    }

    Node affine(const Node& input, const Node& weight, const Node& bias) {
        return {this, make<layer::Affine>(input.get(), weight.get(), bias.get())};
    }

    Node sparse_affine(const Node& input, const Node& weight, const Node& bias) {
        auto* ii = dynamic_cast<layer::InputInt*>(input.get());
        if (!ii)
            error("GraphBuilder: sparse_affine requires InputInt as input");
        return {this, make<layer::SparseAffine>(ii, weight.get(), bias.get())};
    }

    Node select(const Node& input, layer::BucketIndex* index) {
        return {this, make<layer::Select>(input.get(), index)};
    }

    Node pairwise_mul(const Node& input) { return {this, make<layer::PairwiseMul>(input.get())}; }

    Node concat(std::vector<Node> inputs, layer::ConcatAxis axis = layer::ConcatAxis::Rows) {
        std::vector<layer::Layer*> layers;
        for (const auto& n : inputs)
            layers.push_back(n.get());
        return {this, make<layer::Concat>(std::move(layers), axis)};
    }

    Node repeat(const Node& input, int count) {
        return concat(std::vector<Node>(count, input), layer::ConcatAxis::Cols);
    }

    Node mean(const Node& input) { return {this, make<layer::Mean>(input.get())}; }

  private:
    Graph& graph_;
    int affine_param_group_count_ = 0;

    template <typename T, typename... Args>
    layer::Layer* make(Args&&... args) {
        return graph_.emplace<T>(std::forward<Args>(args)...);
    }

    Node affine_unary(const Node& x, float scale, float bias) {
        return {this, make<layer::ElemwiseUnary>(x.get(), kernel::AddScaleUnary{scale, bias})};
    }
};

} // namespace nn::graph
