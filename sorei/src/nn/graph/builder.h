#pragma once

#include <string>
#include <vector>

#include "graph.h"
#include "node.h"

namespace sorei::nn {

class GraphBuilder {
  public:
    GraphBuilder(Graph& graph)
        : graph_(graph) {}

    GraphBuilder(const GraphBuilder&) = delete;
    GraphBuilder& operator=(const GraphBuilder&) = delete;
    GraphBuilder(GraphBuilder&&) = delete;
    GraphBuilder& operator=(GraphBuilder&&) = delete;

    Node input_int(const std::string& name, const matrix::Shape& shape) {
        return {this, graph_.emplace_named<InputInt>(name, shape)};
    }

    Node input_float(const std::string& name, const matrix::Shape& shape) {
        return {this, graph_.emplace_named<InputFloat>(name, shape)};
    }

    BucketIndex* bucket_index(const std::string& name, int count, int size) {
        return graph_.emplace_named<BucketIndex>(name, count, size);
    }

    ParamNode param(int input_dim, int output_dim) {
        return {this, make<Param>(matrix::Shape{output_dim, input_dim})};
    }

    AffineLayer affine_layer(int input_dim, int output_dim) {
        auto w = param(input_dim, output_dim);
        w.he_init(input_dim);
        auto b = param(1, output_dim);
        return {w, b};
    }

    Node relu(const Node& x) { return {this, make<ElemwiseUnary>(x.get(), cuda::ReLU{})}; }

    // relu clamped to [0, 1]
    Node clamped_relu(const Node& x) {
        return {this, make<ElemwiseUnary>(x.get(), cuda::ClampedReLU{})};
    }

    // relu clamped to [0, 1] and squared
    Node squared_clamped_relu(const Node& x) {
        return {this, make<ElemwiseUnary>(x.get(), cuda::SquaredClampedReLU{})};
    }

    Node sigmoid(const Node& x) { return {this, make<ElemwiseUnary>(x.get(), cuda::Sigmoid{})}; }

    Node abs(const Node& x) { return {this, make<ElemwiseUnary>(x.get(), cuda::Abs{})}; }

    Node neg(const Node& x) { return affine_unary(x, -1.0f, 0.0f); }

    Node clamp(const Node& x, float lo, float hi) {
        return {this, make<ElemwiseUnary>(x.get(), cuda::Clamp{lo, hi})};
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
        return {this, make<ElemwiseUnary>(x.get(), cuda::DivLeftUnary{s})};
    }

    Node add(const Node& a, const Node& b) {
        return {this, make<ElemwiseBinary>(a.get(), b.get(), cuda::AddBinary{})};
    }

    Node sub(const Node& a, const Node& b) {
        return {this, make<ElemwiseBinary>(a.get(), b.get(), cuda::SubBinary{})};
    }

    Node mul(const Node& a, const Node& b) {
        return {this, make<ElemwiseBinary>(a.get(), b.get(), cuda::MulBinary{})};
    }

    Node div(const Node& a, const Node& b) {
        return {this, make<ElemwiseBinary>(a.get(), b.get(), cuda::DivBinary{})};
    }

    Node mat_mul(const Node& weight, const Node& input) {
        return {this, make<MatMul>(weight.get(), input.get())};
    }

    Node affine(const Node& input, const Node& weight, const Node& bias) {
        auto* ii = dynamic_cast<InputInt*>(input.get());
        if (ii)
            return {this, make<SparseAffine>(ii, weight.get(), bias.get())};
        else
            return {this, make<Affine>(input.get(), weight.get(), bias.get())};
    }

    Node select(const Node& input, BucketIndex* index) {
        return {this, make<Select>(input.get(), index)};
    }

    Node pairwise_mul(const Node& input) { return {this, make<PairwiseMul>(input.get())}; }

    Node concat(std::vector<Node> inputs, ConcatAxis axis = ConcatAxis::Rows) {
        std::vector<Layer*> layers;
        for (const auto& n : inputs)
            layers.push_back(n.get());
        return {this, make<Concat>(std::move(layers), axis)};
    }

    Node repeat(const Node& input, int count) {
        return concat(std::vector<Node>(count, input), ConcatAxis::Cols);
    }

    Node mean(const Node& input) { return {this, make<Mean>(input.get())}; }

    Node softmax_cross_entropy(const Node& logits, const Node& labels) {
        auto* li = dynamic_cast<InputInt*>(labels.get());
        if (!li)
            error("GraphBuilder: softmax_cross_entropy requires InputInt as labels");
        return {this, make<SoftmaxCrossEntropy>(logits.get(), li)};
    }

  private:
    Graph& graph_;

    template <typename T, typename... Args>
    Layer* make(Args&&... args) {
        return graph_.emplace<T>(std::forward<Args>(args)...);
    }

    Node affine_unary(const Node& x, float scale, float bias) {
        return {this, make<ElemwiseUnary>(x.get(), cuda::AddScaleUnary{scale, bias})};
    }
};

} // namespace sorei::nn
