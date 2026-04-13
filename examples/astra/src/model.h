#pragma once

#include <fstream>

#include "binpack_loader.h"
#include "sorei/nn.h"

namespace {

template <typename T = float>
static void
write_quantized(std::ostream& f, const sorei::tensor::CPUMatrix<float>& src, int scale = 1) {
    sorei::tensor::CPUArray<T> dst(src.size());

    for (int i = 0; i < src.size(); i++) {
        if constexpr (std::is_same_v<T, float>) {
            dst[i] = src(i);
        } else {
            constexpr T lo = std::numeric_limits<T>::min();
            constexpr T hi = std::numeric_limits<T>::max();
            const T qv = static_cast<T>(std::round(src(i) * scale));
            if (qv < lo || qv > hi)
                sorei::println("Warning: value {} out of range, clamping", src(i));
            dst[i] = std::clamp(qv, lo, hi);
        }
    }

    f.write(reinterpret_cast<const char*>(dst.data()), dst.size() * sizeof(T));
    if (!f.good())
        sorei::error("Failed writing quantized data");
}

} // namespace

struct AstraModel : public sorei::nn::Model {
    using AffineLayer = sorei::nn::graph::AffineLayer;
    using TrainingDataEntry = binpack::TrainingDataEntry;

    static constexpr int FT_SIZE = 1024;
    static constexpr int L1_SIZE = 16;
    static constexpr int L2_SIZE = 32;
    static constexpr int OUTPUT_BUCKETS = 8;

    AffineLayer ft, l1, l2, l3;

    sorei::nn::GraphOutput build_graph(sorei::nn::graph::GraphBuilder& b) override {
        // layers
        ft = b.affine_layer(768, FT_SIZE);
        l1 = b.affine_layer(FT_SIZE, L1_SIZE * OUTPUT_BUCKETS);
        l2 = b.affine_layer(2 * L1_SIZE, L2_SIZE * OUTPUT_BUCKETS);
        l3 = b.affine_layer(L2_SIZE, OUTPUT_BUCKETS);

        // inputs
        auto stm_in = b.input_int({32, 0}, "stm_in");
        auto nstm_in = b.input_int({32, 0}, "nstm_in");
        auto output_bucket = b.bucket_index(OUTPUT_BUCKETS, 0, "output_bucket");
        auto target = b.input_float({1, 0}, "target");

        // forward pass
        auto ft_stm = ft(stm_in).clamped_relu().pairwise_mul();
        auto ft_nstm = ft(nstm_in).clamped_relu().pairwise_mul();
        auto cat_ft = b.concat({ft_stm, ft_nstm});

        auto l1_out = l1(cat_ft).select(output_bucket);
        auto cat_l1 = b.concat({l1_out, l1_out * l1_out}).clamped_relu();

        auto l2_out = l2(cat_l1).select(output_bucket).clamped_relu();
        auto l3_out = l3(l2_out).select(output_bucket);

        // loss
        auto diff = l3_out.sigmoid() - target;
        auto loss = (diff * diff).mean();

        return {.prediction = l3_out, .loss = loss};
    }

    float predict(const std::string& fen) {
        chess::Position pos;
        pos.set(fen);
        std::vector<binpack::TrainingDataEntry> ds{{pos}};

        BatchData batch{ds};

        forward({
            {"stm_in", batch.stm_indices()},
            {"nstm_in", batch.nstm_indices()},
            {"output_bucket", batch.bucket_indices()},
            {"target", batch.targets()},
        });

        return prediction().to_cpu()(0) * BatchData::EVAL_SCALE;
    }

    void quantize_params(const std::string& path = "quantized_model.nnue") {
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open())
            sorei::error("Failed writing quantized parameters to {}", path);

        write_quantized<int16_t>(f, ft.weight.data().to_cpu(), 255);
        write_quantized<int16_t>(f, ft.bias.data().to_cpu(), 255);
        write_quantized<int8_t>(f, l1.weight.data().to_cpu().transpose(), 64);
        write_quantized<float>(f, l1.bias.data().to_cpu());
        write_quantized<float>(f, l2.weight.data().to_cpu().transpose());
        write_quantized<float>(f, l2.bias.data().to_cpu());
        write_quantized<float>(f, l3.weight.data().to_cpu().transpose());
        write_quantized<float>(f, l3.bias.data().to_cpu());
    }
};
