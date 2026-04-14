#pragma once

#include <fstream>
#include <initializer_list>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "graph/include.h"
#include "network.h"
#include "tensor.h"

namespace sorei::nn {

struct GraphOutput {
    graph::Node prediction;
    graph::Node loss;
};

class Model {
  public:
    struct InputBinding {
        std::string name;
        std::variant<const Tensor<int>*, const Tensor<float>*> tensor;

        InputBinding(std::string name, const Tensor<int>& t)
            : name(std::move(name)),
              tensor(&t) {}
        InputBinding(std::string name, const Tensor<float>& t)
            : name(std::move(name)),
              tensor(&t) {}
    };

    Model() = default;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = delete;
    Model& operator=(Model&&) = delete;
    virtual ~Model() = default;

    virtual GraphOutput build_graph(graph::GraphBuilder&) = 0;

    void forward(std::initializer_list<InputBinding> inputs) {
        auto& net = network();
        for (const auto& input : inputs)
            std::visit([&](const auto* t) { upload(input.name, *t); }, input.tensor);
        net.forward();
    }

    void backward() { network().backward(); }

    void print_graph() {
        network();
        graph_.print();
    }

    void clear_running_loss() { network().running_loss().clear(); }
    float running_loss() { return network().running_loss().to_host()(0); }

    std::vector<layer::Param*> params() { return network().params(); }
    tensor::DeviceMatrix<float>& prediction() { return network().prediction(); }

    layer::Param& get_param(const std::string& name) { return *graph_.get<layer::Param>(name); }

    void load_params(const std::string& file) {
        std::ifstream f(file, std::ios::binary);
        if (!f)
            error("Model: file {} does not exist", file);

        for (auto* p : net_->params())
            load_param(p->data(), f);
    }

    void save_params(const std::string& file) {
        std::ofstream f(file, std::ios::binary);
        if (!f)
            error("Model: failed writing params to {}", file);

        for (auto* p : net_->params())
            save_param(p->data(), f);
    }

  private:
    graph::Graph graph_;
    std::unique_ptr<network::Network> net_;

    network::Network& network() {
        if (!net_) {
            graph::GraphBuilder b{graph_};
            auto [pred_node, loss_node] = build_graph(b);
            auto* pred = pred_node.get();
            auto* loss = loss_node.get();

            if (loss && loss->shape().rows() != 1)
                error("Model: loss output must be a scalar");

            graph::GraphOptimizer{}.optimize(graph_, pred, loss);
            net_ = std::make_unique<network::Network>(graph_.topological_sort(), pred, loss);
        }
        return *net_;
    }

    layer::Layer* get_layer(const std::string& name) {
        for (const auto& node : graph_.nodes())
            if (node->name() == name)
                return node.get();
        error("Model: input '{}' not found", name);
    }

    template <typename T, typename Dst>
    void upload_best(Dst& dst, const Tensor<T>& t) {
        if (t.has_device_data())
            dst.upload(t.device_data());
        else if (t.has_host_pinned_data())
            dst.upload(t.host_pinned_data());
        else if (t.has_host_data())
            dst.upload(t.host_data());
        else
            error("Tensor has no data to upload");
    }

    void upload(const std::string& name, const Tensor<int>& t) {
        auto* layer = get_layer(name);

        if (auto* s = dynamic_cast<layer::InputInt*>(layer)) {
            s->resize(to_shape(t));
            upload_best(s->data(), t);
        } else if (auto* b = dynamic_cast<layer::BucketIndex*>(layer)) {
            SOREI_CHECK(t.shape().size() == 1);
            b->resize(t.shape()[0]);
            upload_best(b->data(), t);
        } else {
            error("Model: '{}' must map to InputInt or BucketIndex", name);
        }
    }

    void upload(const std::string& name, const Tensor<float>& t) {
        auto* layer = get_layer(name);
        auto* input = dynamic_cast<layer::InputFloat*>(layer);
        if (!input)
            error("Model: '{}' must map to InputFloat", name);
        input->resize(to_shape(t));
        upload_best(input->data(), t);
    }

    template <typename T>
    static tensor::Shape to_shape(const Tensor<T>& t) {
        auto s = t.shape();
        if (s.size() == 1)
            return {1, s[0]};
        if (s.size() == 2)
            return {s[0], s[1]};
        error("to_shape: unsupported tensor dimension");
    }

    static void save_param(const tensor::DeviceMatrix<float>& data, std::ostream& f) {
        auto host = data.to_host();

        f.write(reinterpret_cast<const char*>(host.data()), host.size() * sizeof(float));
        if (!f)
            error("Model: failed writing data to file");
    }

    static void load_param(tensor::DeviceMatrix<float>& data, std::istream& f) {
        tensor::HostMatrix<float> host(data.shape());

        const size_t expected = host.size() * sizeof(float);

        f.read(reinterpret_cast<char*>(host.data()), expected);
        if (static_cast<size_t>(f.gcount()) != expected)
            error("Model: file too short, expected {} elements", host.size());

        data.upload(host);
    }
};

} // namespace sorei::nn
