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

namespace sorei::nn::graph {

class Graph {
  public:
    Graph() = default;

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) = delete;
    Graph& operator=(Graph&&) = delete;

    void print() const {
        auto top_order = topological_sort();

        std::unordered_map<layer::Layer*, int> index;
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
            if (auto* sa = dynamic_cast<layer::SparseAffineBase*>(n))
                if (sa->has_activation())
                    extra += " [+" + layer::elemwise_op_name(sa->activation()) + "]";
            if (auto* spwm = dynamic_cast<layer::SparseAffinePairwiseMul*>(n))
                extra += " [+PairwiseMul]";
            if (auto* c = dynamic_cast<layer::ConcatBase*>(n))
                extra += (c->axis() == layer::ConcatAxis::Rows) ? " [axis=rows]" : " [axis=cols]";

            std::cout << "[" << std::right << std::setw(2) << i << "] " << std::left
                      << std::setw(23) << n->name() << " dim=" << std::setw(4) << n->shape().rows()
                      << (inputs.empty() ? "" : " <- [" + inputs + "]") << extra << "\n";
        }
    }

    std::vector<layer::Layer*> topological_sort() const {
        std::vector<layer::Layer*> order;
        std::unordered_set<layer::Layer*> visited;

        std::function<void(layer::Layer*)> dfs = [&](layer::Layer* node) {
            if (!node || visited.count(node))
                return;
            visited.insert(node);
            for (layer::Layer* dep : node->inputs())
                dfs(dep);
            order.push_back(node);
        };

        for (const auto& node : nodes_)
            dfs(node.get());

        return order;
    }

    const std::vector<std::unique_ptr<layer::Layer>>& nodes() const { return nodes_; }
    std::size_t size() const { return nodes_.size(); }

  private:
    std::vector<std::unique_ptr<layer::Layer>> nodes_;
    std::unordered_map<std::string, layer::Layer*> named_ops_;

    friend class GraphBuilder;
    friend class GraphOptimizer;

    template <typename T, typename... Args>
    layer::Layer* emplace(Args&&... args) {
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

    void erase(layer::Layer* op) {
        SOREI_CHECK(op);
        auto it = std::find_if(nodes_.begin(), nodes_.end(), [op](const auto& p) {
            return p.get() == op;
        });

        if (it != nodes_.end()) {
            for (auto nit = named_ops_.begin(); nit != named_ops_.end(); ++nit) {
                if (nit->second == op) {
                    named_ops_.erase(nit);
                    break;
                }
            }
            nodes_.erase(it);
        }
    }
};

} // namespace sorei::nn::graph
