#pragma once

#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../layer/include.h"

namespace sorei::nn {

class Graph {
  public:
    Graph() = default;

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) = delete;
    Graph& operator=(Graph&&) = delete;

    void print(const std::vector<Layer*>& output_roots) const {
        auto top_order = topological_sort(output_roots);

        std::unordered_map<Layer*, int> index;
        for (int i = 0; i < (int)top_order.size(); i++)
            index[top_order[i]] = i;

        for (int i = 0; i < (int)top_order.size(); i++) {
            auto* n = top_order[i];

            std::string inputs;
            for (auto* inp : n->inputs()) {
                if (!inputs.empty())
                    inputs += ", ";
                inputs += std::to_string(index[inp]);
            }

            std::string extra;
            if (auto* sa = dynamic_cast<SparseAffineBase*>(n))
                if (sa->has_activation())
                    extra += " [+" + elemwise_op_name(sa->activation()) + "]";
            if (auto* spwm = dynamic_cast<SparseAffinePairwiseMul*>(n))
                extra += " [+PairwiseMul]";
            if (auto* c = dynamic_cast<ConcatBase*>(n))
                extra += (c->axis() == ConcatAxis::Rows) ? " [axis=rows]" : " [axis=cols]";

            std::cout << "[" << std::right << std::setw(2) << i << "] " << std::left
                      << std::setw(23) << n->name() << " dim=" << std::setw(4) << n->shape().rows()
                      << (inputs.empty() ? "" : " <- [" + inputs + "]") << extra << "\n";
        }
    }

    std::vector<Layer*> topological_sort(const std::vector<Layer*>& output_roots) const {
        std::vector<Layer*> order;
        std::unordered_set<Layer*> visited;
        std::unordered_set<Layer*> in_stack;

        std::function<void(Layer*)> dfs = [&](Layer* node) {
            if (!node || visited.count(node))
                return;
            if (in_stack.count(node))
                error("Graph: cycle detected");

            in_stack.insert(node);
            for (Layer* dep : node->inputs())
                dfs(dep);
            in_stack.erase(node);

            visited.insert(node);
            order.push_back(node);
        };

        for (Layer* root : output_roots)
            dfs(root);

        return order;
    }

    const std::vector<std::unique_ptr<Layer>>& nodes() const { return nodes_; }
    std::size_t size() const { return nodes_.size(); }

  private:
    std::vector<std::unique_ptr<Layer>> nodes_;
    std::unordered_map<std::string, Layer*> named_ops_;

    friend class GraphBuilder;
    friend class GraphOptimizer;

    template <typename T, typename... Args>
    Layer* emplace(Args&&... args) {
        nodes_.push_back(std::make_unique<T>(std::forward<Args>(args)...));
        SOREI_CHECK(nodes_.back());
        return nodes_.back().get();
    }

    template <typename T, typename... Args>
    T* emplace_named(const std::string& name, Args&&... args) {
        auto [it, inserted] = named_ops_.try_emplace(name, nullptr);
        if (!inserted)
            error("Graph: duplicate name '{}'", name);

        it->second = emplace<T>(std::forward<Args>(args)..., name);
        return static_cast<T*>(it->second);
    }

    void erase(Layer* op) {
        SOREI_CHECK(op);
        auto it = std::find_if(nodes_.begin(), nodes_.end(), [op](const auto& p) {
            return p.get() == op;
        });

        if (it != nodes_.end()) {
            std::erase_if(named_ops_, [op](const auto& kv) { return kv.second == op; });
            nodes_.erase(it);
        }
    }
};

} // namespace sorei::nn
