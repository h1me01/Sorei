#include "builder.h"

namespace nn::graph {

// Node

Node Node::relu() const { return gb().relu(*this); }
Node Node::clamped_relu() const { return gb().clamped_relu(*this); }
Node Node::squared_clamped_relu() const { return gb().squared_clamped_relu(*this); }
Node Node::sigmoid() const { return gb().sigmoid(*this); }
Node Node::abs() const { return gb().abs(*this); }
Node Node::neg() const { return gb().neg(*this); }

Node Node::pairwise_mul() const { return gb().pairwise_mul(*this); }
Node Node::mean() const { return gb().mean(*this); }
Node Node::pow(float e) const { return gb().pow(*this, e); }
Node Node::clamp(float lo, float hi) const { return gb().clamp(*this, lo, hi); }
Node Node::repeat(int count) const { return gb().repeat(*this, count); }
Node Node::select(layer::BucketIndex* index) const { return gb().select(*this, index); }

Node Node::operator+(const Node& rhs) const { return gb().add(*this, rhs); }
Node Node::operator-(const Node& rhs) const { return gb().sub(*this, rhs); }
Node Node::operator*(const Node& rhs) const { return gb().mul(*this, rhs); }
Node Node::operator/(const Node& rhs) const { return gb().div(*this, rhs); }

Node Node::operator+(float s) const { return gb().add(*this, s); }
Node Node::operator-(float s) const { return gb().sub(*this, s); }
Node Node::operator*(float s) const { return gb().mul(*this, s); }
Node Node::operator/(float s) const { return gb().div(*this, s); }

Node operator+(float s, const Node& n) { return n.gb().add(n, s); }
Node operator-(float s, const Node& n) { return n.gb().sub(s, n); }
Node operator*(float s, const Node& n) { return n.gb().mul(n, s); }
Node operator/(float s, const Node& n) { return n.gb().div(s, n); }

// AffineLayer

Node AffineLayer::operator()(const Node& input) const {
    return weight.gb().affine(input, weight, bias);
}

Node AffineLayer::operator()(layer::SparseInput* input) const {
    return weight.gb().sparse_affine(input, weight, bias);
}

} // namespace nn::graph
