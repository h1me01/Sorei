#pragma once

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

    // Fusion passes

    void fuse_sparse_affine() {
        // fuse activation into SparseAffine
        fixed_point<SparseAffine>([this](SparseAffine& sa, const ConsumerMap& consumers) -> bool {
            auto* unary = dynamic_cast<ElemwiseUnary*>(sole_consumer(&sa, consumers));
            if (!unary)
                return false;

            if (!sa.set_activation(unary->op()))
                return false;

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
                static_cast<SparseAffineBase*>(inp)->fuse(fused);

            redirect_and_remove(&cn, fused);
            return true;
        });
    }
};

} // namespace sorei::nn
