#pragma once

#include <optional>
#include <vector>

#include "graph.h"

namespace sorei::nn {

class GraphOptimizer {
  public:
    GraphOptimizer(Graph& graph, Layer*& prediction, Layer*& loss)
        : graph_(graph),
          prediction_(prediction),
          loss_(loss) {

        fuse_sparse_affine();
        fuse_concat();

        prediction = prediction_;
        loss = loss_;
    }

  private:
    Graph& graph_;
    Layer* prediction_ = nullptr;
    Layer* loss_ = nullptr;

    using ConsumerMap = std::unordered_map<Layer*, std::vector<Layer*>>;

    static Layer* sole_consumer(Layer* op, const ConsumerMap& consumers) {
        auto it = consumers.find(op);
        if (it == consumers.end() || it->second.size() != 1)
            return nullptr;
        return it->second[0];
    }

    void redirect_and_remove(Layer* removed, Layer* replacement) {
        for (auto* node : graph_.topological_sort({prediction_, loss_}))
            if (node != removed)
                node->replace_input(removed, replacement);

        if (prediction_ == removed)
            prediction_ = replacement;
        if (loss_ == removed)
            loss_ = replacement;

        graph_.erase(removed);
    }

    template <typename T, typename Action>
    void fixed_point(Action action) {
        bool changed = true;
        while (changed) {
            changed = false;

            ConsumerMap consumers;
            for (auto* node : graph_.topological_sort({prediction_, loss_})) {
                consumers[node];
                for (auto* inp : node->inputs())
                    consumers[inp].push_back(node);
            }

            for (auto* node : graph_.topological_sort({prediction_, loss_})) {
                if (auto* t = dynamic_cast<T*>(node)) {
                    if (action(*t, consumers)) {
                        changed = true;
                        break;
                    }
                }
            }
        }
    }

    static std::optional<ActOp> as_activation(const ElemwiseUnary::Op& op) {
        if (std::holds_alternative<unary::ReLU>(op))
            return ActOp{std::get<unary::ReLU>(op)};
        if (std::holds_alternative<unary::ClampedReLU>(op))
            return ActOp{std::get<unary::ClampedReLU>(op)};
        if (std::holds_alternative<unary::SquaredClampedReLU>(op))
            return ActOp{std::get<unary::SquaredClampedReLU>(op)};
        return std::nullopt;
    }

    // Fusion passes

    void fuse_sparse_affine() {
        // fuse activation into SparseAffine
        fixed_point<SparseAffine>([this](SparseAffine& sa, const ConsumerMap& consumers) -> bool {
            auto* unary = dynamic_cast<ElemwiseUnary*>(sole_consumer(&sa, consumers));
            if (!unary)
                return false;

            auto act = as_activation(unary->op());
            if (!act)
                return false;

            sa.set_activation(*act);
            redirect_and_remove(unary, &sa);
            return true;
        });

        // fuse pairwise-mul into SparseAffinePairwiseMul
        fixed_point<SparseAffine>([this](SparseAffine& sa, const ConsumerMap& consumers) -> bool {
            auto* pw = dynamic_cast<PairwiseMul*>(sole_consumer(&sa, consumers));
            if (!pw)
                return false;

            auto* fused = static_cast<SparseAffinePairwiseMul*>(
                graph_.emplace<SparseAffinePairwiseMul>(sa.input(), sa.weight(), sa.bias())
            );
            fused->set_activation(sa.activation());

            redirect_and_remove(&sa, fused);
            redirect_and_remove(pw, fused);
            return true;
        });
    }

    void fuse_concat() {
        // fuse SparseAffineBase with row-wise FusedConcat
        fixed_point<Concat>([this](Concat& cn, const ConsumerMap& consumers) -> bool {
            if (cn.axis() != ConcatAxis::Rows)
                return false;

            bool ok = std::ranges::all_of(cn.inputs(), [&](Layer* inp) {
                return sole_consumer(inp, consumers) == &cn && dynamic_cast<SparseAffineBase*>(inp);
            });
            if (!ok)
                return false;

            auto* fused = static_cast<FusedConcat*>(graph_.emplace<FusedConcat>(cn.inputs()));
            for (auto* inp : fused->inputs())
                static_cast<SparseAffineBase*>(inp)->fuse_with_concat(fused);

            redirect_and_remove(&cn, fused);
            return true;
        });

        // fuse activation following FusedConcat into each SparseAffineBase input
        fixed_point<FusedConcat>([this](FusedConcat& cn, const ConsumerMap& consumers) -> bool {
            auto* unary = dynamic_cast<ElemwiseUnary*>(sole_consumer(&cn, consumers));
            if (!unary)
                return false;

            auto act = as_activation(unary->op());
            if (!act)
                return false;

            bool ok = std::ranges::all_of(cn.inputs(), [&](Layer* inp) {
                return !static_cast<SparseAffineBase*>(inp)->has_activation();
            });
            if (!ok)
                return false;

            for (auto* inp : cn.inputs())
                static_cast<SparseAffineBase*>(inp)->set_activation(*act);

            redirect_and_remove(unary, &cn);
            return true;
        });
    }
};

} // namespace sorei::nn
