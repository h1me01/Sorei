#pragma once

#include <fstream>
#include <initializer_list>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "../matrix/include.h"
#include "graph/include.h"
#include "network.h"

namespace sorei::nn {

struct GraphOutput {
    Node prediction;
    Node loss;
};

class Model {
  public:
    struct InputBinding {
        std::string name;
        std::variant<
            const matrix::HostMatrix<int>*,
            const matrix::HostMatrix<float>*,
            const matrix::HostPinnedMatrix<int>*,
            const matrix::HostPinnedMatrix<float>*,
            const matrix::DeviceMatrix<int>*,
            const matrix::DeviceMatrix<float>*>
            matrix;

        template <typename T>
            requires std::constructible_from<decltype(matrix), const T*>
        InputBinding(std::string name, const T& m)
            : name(std::move(name)),
              matrix(&m) {}
    };

    Model() = default;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = delete;
    Model& operator=(Model&&) = delete;
    virtual ~Model() = default;

    virtual GraphOutput build_graph(GraphBuilder&) = 0;

    void forward(std::initializer_list<InputBinding> inputs) {
        auto& net = network();
        for (const auto& input : inputs) {
            std::visit(
                [&](const auto* t) {
                    auto it = layer_map_.find(input.name);
                    if (it == layer_map_.end())
                        error("Model: input '{}' not found", input.name);
                    auto* layer = it->second;

                    if (auto* in = dynamic_cast<InputInt*>(layer)) {
                        in->resize(t->shape());
                        in->data().upload(*t);
                    } else if (auto* in = dynamic_cast<InputFloat*>(layer)) {
                        in->resize(t->shape());
                        in->data().upload(*t);
                    } else if (auto* b = dynamic_cast<BucketIndex*>(layer)) {
                        SOREI_CHECK(t->shape().rows() == 1);
                        b->resize(t->size());
                        b->data().upload(*t);
                    } else {
                        error(
                            "Model: '{}' must map to InputInt, InputFloat, or BucketIndex",
                            input.name
                        );
                    }
                },
                input.matrix
            );
        }
        net.forward();
    }

    void backward() { network().backward(); }

    void zero_running_loss() { network().running_loss().clear(); }
    float running_loss() { return network().running_loss().to_host()(0); }

    std::vector<Param*> params() { return network().params(); }
    matrix::DeviceMatrix<float>& prediction() { return network().prediction(); }

    void load_params(const std::string& file) {
        std::ifstream f(file, std::ios::binary);
        if (!f)
            error("Model: file {} does not exist", file);

        for (auto* p : network().params()) {
            matrix::HostMatrix<float> host(p->data().shape());
            const size_t expected = host.size() * sizeof(float);
            f.read(reinterpret_cast<char*>(host.data()), expected);
            if (static_cast<size_t>(f.gcount()) != expected)
                error("Model: file too short, expected {} elements", host.size());
            p->data().upload(host);
        }

        if (f.peek() != EOF)
            error("Model: file {} has extra data after expected parameters", file);
    }

    void save_params(const std::string& file) {
        std::ofstream f(file, std::ios::binary);
        if (!f)
            error("Model: failed writing params to {}", file);

        for (auto* p : network().params()) {
            auto host = p->data().to_host();
            f.write(reinterpret_cast<const char*>(host.data()), host.size() * sizeof(float));
            if (!f)
                error("Model: failed writing data to file");
        }
    }

  private:
    Graph graph_;
    std::unique_ptr<Network> net_;
    std::unordered_map<std::string, Layer*> layer_map_;

    Network& network() {
        if (!net_) {
            GraphBuilder b{graph_};
            auto [pred_node, loss_node] = build_graph(b);
            auto* pred = pred_node.get();
            auto* loss = loss_node.get();

            if (loss && loss->shape().rows() != 1)
                error("Model: loss output must be a scalar");

            GraphOptimizer{graph_, pred, loss};
            auto sorted = graph_.topological_sort({pred, loss});

            for (const auto& node : sorted)
                layer_map_[node->name()] = node;

            net_ = std::make_unique<Network>(std::move(sorted), pred, loss);
        }
        return *net_;
    }
};

} // namespace sorei::nn
