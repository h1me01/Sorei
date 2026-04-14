#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "sorei/nn.h"

#include "framework.h"

namespace sorei::test {

// Numerically verifies backward() for `layer` w.r.t. `params`.
// Returns the max relative error across all parameters and elements.
inline float grad_check(
    nn::layer::TypedLayer<float>* layer,
    std::vector<nn::layer::Param*> params,
    float eps = 1e-3f,
    float* out_abs = nullptr
) {
    using namespace tensor;
    using namespace nn::layer;

    layer->forward();
    cudaDeviceSynchronize();

    cuda::set(layer->grad(), 1.0f);
    cudaDeviceSynchronize();

    for (auto* p : params) {
        p->grad().clear();
        cudaDeviceSynchronize();
    }

    layer->backward();
    cudaDeviceSynchronize();

    struct ParamGrads {
        HostMatrix<float> data;
        HostMatrix<float> grad;
    };
    std::vector<ParamGrads> saved;
    saved.reserve(params.size());
    for (auto* p : params)
        saved.push_back({p->data().to_host(), p->grad().to_host()});

    float max_abs = 0.0f;
    float max_rel = 0.0f;

    for (int pi = 0; pi < (int)params.size(); ++pi) {
        auto* p = params[pi];
        auto& dev_data = p->data();
        const auto& host_data = saved[pi].data;
        const auto& host_agrad = saved[pi].grad;

        for (int i = 0; i < dev_data.size(); ++i) {
            auto host_p = host_data;
            host_p(i) += eps;
            dev_data.upload(host_p);
            layer->forward();
            cudaDeviceSynchronize();
            const auto out_p = layer->data().to_host();

            auto host_m = host_data;
            host_m(i) -= eps;
            dev_data.upload(host_m);
            layer->forward();
            cudaDeviceSynchronize();
            const auto out_m = layer->data().to_host();

            float num = 0.0f;
            for (int j = 0; j < out_p.size(); ++j)
                num += out_p(j) - out_m(j);
            num /= 2.0f * eps;

            const float ana = host_agrad(i);
            const float abs_err = std::abs(num - ana);
            const float rel_err = abs_err / (std::abs(num) + std::abs(ana) + 1e-8f);

            max_abs = std::max(max_abs, abs_err);
            max_rel = std::max(max_rel, rel_err);
        }

        dev_data.upload(host_data);
    }

    layer->forward();
    cudaDeviceSynchronize();

    if (out_abs)
        *out_abs = max_abs;
    return max_rel;
}

inline std::unique_ptr<nn::layer::Param>
make_param(int rows, int cols, float lo = -0.5f, float hi = 0.5f) {
    auto p = std::make_unique<nn::layer::Param>(tensor::Shape(rows, cols));
    p->uniform_init(lo, hi);
    return p;
}

inline std::unique_ptr<nn::layer::Param> make_param_filled(int rows, int cols, float val) {
    using namespace tensor;
    auto p = std::make_unique<nn::layer::Param>(Shape(rows, cols));
    HostMatrix<float> m(Shape(rows, cols));
    m.fill(val);
    p->data().upload(m);
    p->grad().clear();
    return p;
}

inline std::unique_ptr<nn::layer::InputInt> make_input_int(int cols, const std::vector<int>& vals) {
    using namespace tensor;
    auto inp = std::make_unique<nn::layer::Input<int>>(Shape(1, cols));
    HostMatrix<int> m(Shape(1, cols));
    for (int i = 0; i < cols; ++i)
        m(0, i) = vals[i];
    inp->data().upload(m);
    return inp;
}

} // namespace sorei::test
