#pragma once

#include <functional>
#include <optional>
#include <vector>

#include "graph.h"

namespace sorei::nn::graph {

class GraphOptimizer {
  public:
    void optimize(Graph& graph, layer::Layer*& prediction, layer::Layer*& loss) {
        prediction_ = prediction;
        loss_ = loss;

        fold_self_mul(graph);
        fuse_sparse_affine(graph);
        fuse_concat(graph);

        prediction = prediction_;
        loss = loss_;
    }

  private:
    layer::Layer* prediction_ = nullptr;
    layer::Layer* loss_ = nullptr;

    // Consumer map

    using ConsumerMap = std::unordered_map<layer::Layer*, std::vector<layer::Layer*>>;

    static ConsumerMap build_consumers(const Graph& graph) {
        ConsumerMap c;
        for (const auto& node : graph.nodes()) {
            c[node.get()];
            for (auto* inp : node->inputs())
                c[inp].push_back(node.get());
        }
        return c;
    }

    static layer::Layer* sole_consumer(layer::Layer* op, const ConsumerMap& consumers) {
        auto it = consumers.find(op);
        if (it == consumers.end() || it->second.size() != 1)
            return nullptr;
        return it->second[0];
    }

    // Helper

    void redirect_and_remove(Graph& graph, layer::Layer* removed, layer::Layer* replacement) {
        for (const auto& node : graph.nodes())
            if (node.get() != removed)
                node->replace_input(removed, replacement);

        if (prediction_ == removed)
            prediction_ = replacement;
        if (loss_ == removed)
            loss_ = replacement;

        graph.erase(removed);
    }

    template <typename T>
    void fixed_point(Graph& graph, const std::function<bool(Graph&, T*)>& action) {
        bool changed = true;
        while (changed) {
            changed = false;
            std::vector<layer::Layer*> snapshot;
            for (const auto& n : graph.nodes())
                snapshot.push_back(n.get());

            for (auto* raw : snapshot) {
                if (auto* t = dynamic_cast<T*>(raw)) {
                    if (action(graph, t)) {
                        changed = true;
                        break;
                    }
                }
            }
        }
    }

    static std::optional<layer::ActOp> as_activation(const layer::ElemwiseUnary::Op& op) {
        if (std::holds_alternative<cuda::ReLU>(op))
            return layer::ActOp{std::get<cuda::ReLU>(op)};
        if (std::holds_alternative<cuda::ClampedReLU>(op))
            return layer::ActOp{std::get<cuda::ClampedReLU>(op)};
        if (std::holds_alternative<cuda::SquaredClampedReLU>(op))
            return layer::ActOp{std::get<cuda::SquaredClampedReLU>(op)};
        return std::nullopt;
    }

    // Folding passes

    void fold_self_mul(Graph& graph) {
        fixed_point<layer::ElemwiseBinary>(
            graph,
            [this](Graph& g, layer::ElemwiseBinary* eb) -> bool {
                if (eb->name() != "Mul")
                    return false;

                auto ins = eb->inputs();
                if (ins[0] != ins[1])
                    return false;

                auto* replacement = g.emplace<layer::ElemwiseUnary>(ins[0], cuda::PowInt{2});
                redirect_and_remove(g, eb, replacement);
                return true;
            }
        );
    }

    // Fusion passes

    void fuse_sparse_affine(Graph& graph) {
        // fuse activation into SparseAffine
        fixed_point<layer::SparseAffine>(graph, [this](Graph& g, layer::SparseAffine* sa) -> bool {
            auto consumers = build_consumers(g);

            auto* unary = dynamic_cast<layer::ElemwiseUnary*>(sole_consumer(sa, consumers));
            if (!unary)
                return false;

            auto act = as_activation(unary->op());
            if (!act)
                return false;

            sa->set_activation(*act);
            redirect_and_remove(g, unary, sa);

            return true;
        });

        // fuse pairwise-mul into SparseAffinePairwiseMul
        fixed_point<layer::SparseAffine>(graph, [this](Graph& g, layer::SparseAffine* sa) -> bool {
            auto consumers = build_consumers(g);

            auto* pw = dynamic_cast<layer::PairwiseMul*>(sole_consumer(sa, consumers));
            if (!pw)
                return false;

            auto* fused = static_cast<layer::SparseAffinePairwiseMul*>(
                g.emplace<layer::SparseAffinePairwiseMul>(sa->input(), sa->weight(), sa->bias())
            );
            fused->set_activation(sa->activation());

            redirect_and_remove(g, sa, fused);
            redirect_and_remove(g, pw, fused);

            return true;
        });
    }

    void fuse_concat(Graph& graph) {
        // fuse SparseAffineBase with row-wise FusedConcat
        fixed_point<layer::Concat>(graph, [this](Graph& g, layer::Concat* cn) -> bool {
            if (cn->axis() != layer::ConcatAxis::Rows)
                return false;

            auto consumers = build_consumers(g);
            bool ok = std::ranges::all_of(cn->inputs(), [&](layer::Layer* inp) {
                return sole_consumer(inp, consumers) == cn &&
                       dynamic_cast<layer::SparseAffineBase*>(inp);
            });
            if (!ok)
                return false;

            auto* fused =
                static_cast<layer::FusedConcat*>(g.emplace<layer::FusedConcat>(cn->inputs()));
            for (auto* inp : fused->inputs())
                static_cast<layer::SparseAffineBase*>(inp)->fuse_with_concat(fused);

            redirect_and_remove(g, cn, fused);
            return true;
        });

        // fuse activation following FusedConcat into each SparseAffineBase input
        fixed_point<layer::FusedConcat>(graph, [this](Graph& g, layer::FusedConcat* cn) -> bool {
            auto consumers = build_consumers(g);

            auto* unary = dynamic_cast<layer::ElemwiseUnary*>(sole_consumer(cn, consumers));
            if (!unary)
                return false;

            auto act = as_activation(unary->op());
            if (!act)
                return false;

            bool valid = std::ranges::all_of(cn->inputs(), [&](layer::Layer* inp) {
                auto* sa = dynamic_cast<layer::SparseAffineBase*>(inp);
                return sole_consumer(inp, consumers) == cn && sa && !sa->has_activation();
            });
            if (!valid)
                return false;

            for (auto* inp : cn->inputs())
                static_cast<layer::SparseAffineBase*>(inp)->set_activation(*act);

            redirect_and_remove(g, unary, cn);
            return true;
        });
    }
};

} // namespace sorei::nn::graph
