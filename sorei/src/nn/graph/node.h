#pragma once

#include "../../matrix/include.h"
#include "../layer/include.h"

namespace sorei::nn {

class GraphBuilder;

class Node {
  public:
    Node() = default;

    Node(GraphBuilder* gb, Layer* op)
        : gb_(gb),
          layer_(op) {
        SOREI_CHECK(layer_);
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
    Node clamp(float lo, float hi) const;
    Node select(BucketIndex* index) const;
    Node mat_mul(const Node& weight) const;
    Node affine(const Node& weight, const Node& bias) const;
    Node softmax_cross_entropy(const Node& labels) const;

    Node operator+(const Node& rhs) const;
    Node operator-(const Node& rhs) const;
    Node operator*(const Node& rhs) const;
    Node operator/(const Node& rhs) const;

    Node operator+(float s) const;
    Node operator-(float s) const;
    Node operator*(float s) const;
    Node operator/(float s) const;

    GraphBuilder& gb() const {
        SOREI_CHECK(gb_);
        return *gb_;
    }

    virtual Layer* get() const { return layer_; }
    explicit operator bool() const { return layer_ != nullptr; }

  protected:
    GraphBuilder* gb_ = nullptr;
    Layer* layer_ = nullptr;
};

Node operator+(float s, const Node& n);
Node operator-(float s, const Node& n);
Node operator*(float s, const Node& n);
Node operator/(float s, const Node& n);

struct ParamNode : Node {
    ParamNode() = default;

    ParamNode(GraphBuilder* gb, Layer* op)
        : Node(gb, op) {}

    void uniform_init(float lo, float hi) { get()->uniform_init(lo, hi); }
    void he_init(int input_dim) { get()->he_init(input_dim); }
    void set_bounds(float lo, float hi) { get()->set_bounds(lo, hi); }

    matrix::DeviceMatrix<float>& data() { return get()->data(); }
    const matrix::DeviceMatrix<float>& data() const { return get()->data(); }

    int input_dim() const { return get()->data().shape().cols(); }
    int output_dim() const { return get()->data().shape().rows(); }

    Param* get() const { return checked_cast<Param>(Node::get()); }
};

struct AffineLayer {
    ParamNode weight;
    ParamNode bias;

    AffineLayer() = default;

    AffineLayer(ParamNode w, ParamNode b)
        : weight(std::move(w)),
          bias(std::move(b)) {}

    Node operator()(const Node& input) const;
};

} // namespace sorei::nn
