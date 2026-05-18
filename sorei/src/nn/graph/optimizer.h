#pragma once

#include <optional>
#include <vector>

#include "graph.h"

namespace sorei::nn {

class GraphOptimizer {
  public:
    GraphOptimizer(Graph& graph, Layer*& prediction, Layer*& loss) {
        prediction_ = prediction;
        loss_ = loss;

        fuse_sparse_affine(graph);
        fuse_concat(graph);

        prediction = prediction_;
        loss = loss_;
    }

  private:
    Layer* prediction_ = nullptr;
    Layer* loss_ = nullptr;

    using ConsumerMap = std::unordered_map<Layer*, std::vector<Layer*>>;

    ConsumerMap build_consumers(const Graph& graph) {
        ConsumerMap c;
        for (const auto& node : graph.topological_sort({prediction_, loss_})) {
            c[node];
            for (auto* inp : node->inputs())
                c[inp].push_back(node);
        }
        return c;
    }

    static Layer* sole_consumer(Layer* op, const ConsumerMap& consumers) {
        auto it = consumers.find(op);
        if (it == consumers.end() || it->second.size() != 1)
            return nullptr;
        return it->second[0];
    }

    void redirect_and_remove(Graph& graph, Layer* removed, Layer* replacement) {
        for (const auto& node : graph.topological_sort({prediction_, loss_}))
            if (node != removed)
                node->replace_input(removed, replacement);

        if (prediction_ == removed)
            prediction_ = replacement;
        if (loss_ == removed)
            loss_ = replacement;

        graph.erase(removed);
    }

    template <typename T, typename Action>
    void fixed_point(Graph& graph, Action action) {
        bool changed = true;
        while (changed) {
            changed = false;
            const auto consumers = build_consumers(graph);
            std::vector<Layer*> snapshot;
            for (const auto& n : graph.topological_sort({prediction_, loss_}))
                snapshot.push_back(n);

            for (auto* raw : snapshot) {
                if (auto* t = dynamic_cast<T*>(raw)) {
                    if (action(graph, *t, consumers)) {
                        changed = true;
                        break;
                    }
                }
            }
        }
    }

    static std::optional<ActOp> as_activation(const ElemwiseUnary::Op& op) {
        if (std::holds_alternative<cuda::ReLU>(op))
            return ActOp{std::get<cuda::ReLU>(op)};
        if (std::holds_alternative<cuda::ClampedReLU>(op))
            return ActOp{std::get<cuda::ClampedReLU>(op)};
        if (std::holds_alternative<cuda::SquaredClampedReLU>(op))
            return ActOp{std::get<cuda::SquaredClampedReLU>(op)};
        return std::nullopt;
    }

    // Fusion passes

    void fuse_sparse_affine(Graph& graph) {
        // fuse activation into SparseAffine
        fixed_point<SparseAffine>(
            graph,
            [this](Graph& g, SparseAffine& sa, const ConsumerMap& consumers) -> bool {
                auto* unary = dynamic_cast<ElemwiseUnary*>(sole_consumer(&sa, consumers));
                if (!unary)
                    return false;

                auto act = as_activation(unary->op());
                if (!act)
                    return false;

                sa.set_activation(*act);
                redirect_and_remove(g, unary, &sa);
                return true;
            }
        );

        // fuse pairwise-mul into SparseAffinePairwiseMul
        fixed_point<SparseAffine>(
            graph,
            [this](Graph& g, SparseAffine& sa, const ConsumerMap& consumers) -> bool {
                auto* pw = dynamic_cast<PairwiseMul*>(sole_consumer(&sa, consumers));
                if (!pw)
                    return false;

                auto* fused = static_cast<SparseAffinePairwiseMul*>(
                    g.emplace<SparseAffinePairwiseMul>(sa.input(), sa.weight(), sa.bias())
                );
                fused->set_activation(sa.activation());

                redirect_and_remove(g, &sa, fused);
                redirect_and_remove(g, pw, fused);
                return true;
            }
        );
    }

    void fuse_concat(Graph& graph) {
        // fuse SparseAffineBase with row-wise FusedConcat
        fixed_point<Concat>(
            graph,
            [this](Graph& g, Concat& cn, const ConsumerMap& consumers) -> bool {
                if (cn.axis() != ConcatAxis::Rows)
                    return false;

                bool ok = std::ranges::all_of(cn.inputs(), [&](Layer* inp) {
                    return sole_consumer(inp, consumers) == &cn &&
                           dynamic_cast<SparseAffineBase*>(inp);
                });
                if (!ok)
                    return false;

                auto* fused = static_cast<FusedConcat*>(g.emplace<FusedConcat>(cn.inputs()));
                for (auto* inp : fused->inputs())
                    static_cast<SparseAffineBase*>(inp)->fuse_with_concat(fused);

                redirect_and_remove(g, &cn, fused);
                return true;
            }
        );

        // fuse activation following FusedConcat into each SparseAffineBase input
        fixed_point<FusedConcat>(
            graph,
            [this](Graph& g, FusedConcat& cn, const ConsumerMap& consumers) -> bool {
                auto* unary = dynamic_cast<ElemwiseUnary*>(sole_consumer(&cn, consumers));
                if (!unary)
                    return false;

                auto act = as_activation(unary->op());
                if (!act)
                    return false;

                bool valid = std::ranges::all_of(cn.inputs(), [&](Layer* inp) {
                    auto* sa = dynamic_cast<SparseAffineBase*>(inp);
                    return sole_consumer(inp, consumers) == &cn && sa && !sa->has_activation();
                });
                if (!valid)
                    return false;

                for (auto* inp : cn.inputs())
                    static_cast<SparseAffineBase*>(inp)->set_activation(*act);

                redirect_and_remove(g, unary, &cn);
                return true;
            }
        );
    }
};

} // namespace sorei::nn
