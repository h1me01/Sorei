#pragma once

#include <fstream>

#include "sf_binpack/training_data_format.h"
#include "sorei/nn.h"

namespace {

static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static int feature_index(chess::Piece pc, chess::Square psq, chess::Color view) {
    if (view == chess::Color::Black)
        psq.flipVertically();

    return (int)psq + (int)pc.type() * 64 + (pc.color() != view) * 64 * 6;
}

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
                println("Warning: value {} out of range, clamping", src(i));
            dst[i] = std::clamp(qv, lo, hi);
        }
    }

    f.write(reinterpret_cast<const char*>(dst.data()), dst.size() * sizeof(T));
    if (!f.good())
        error("Failed writing quantized data");
}

} // namespace

class AstraModel : public sorei::nn::Model {
    using AffineLayer = sorei::nn::graph::AffineLayer;
    using TrainingDataEntry = binpack::TrainingDataEntry;

  public:
    static constexpr int FT_SIZE = 1024;
    static constexpr int L1_SIZE = 16;
    static constexpr int L2_SIZE = 32;
    static constexpr int OUTPUT_BUCKETS = 8;
    static constexpr float EVAL_SCALE = 400.0f;
    static constexpr float WDL_WEIGHT = 0.7f;

    AffineLayer ft, l1, l2, l3;

    void feed(const std::vector<TrainingDataEntry>& batch) {
        const int batch_size = batch.size();

        stm_indices.resize({32, batch_size});
        nstm_indices.resize({32, batch_size});
        bucket_indices.resize({batch_size});
        targets.resize({batch_size});

        stm_indices.fill(-1);
        nstm_indices.fill(-1);

        for (size_t i = 0; i < batch.size(); i++) {
            const auto& entry = batch[i];
            const chess::Color stm = entry.pos.sideToMove();

            int j = 0;
            for (auto sq : entry.pos.piecesBB()) {
                chess::Piece pc = entry.pos.pieceAt(sq);
                stm_indices(j, i) = feature_index(pc, sq, stm);
                nstm_indices(j, i) = feature_index(pc, sq, !stm);
                ++j;
            }

            const float score_target = sigmoid(entry.score / EVAL_SCALE);
            const float wdl_target = (entry.result + 1) / 2.0f;
            targets[i] = WDL_WEIGHT * wdl_target + (1.0f - WDL_WEIGHT) * score_target;

            bucket_indices[i] = (entry.pos.pieceCount() - 2) / 4;
        }

        forward({
            {"stm_in", stm_indices},
            {"nstm_in", nstm_indices},
            {"output_bucket", bucket_indices},
            {"target", targets},
        });
    }

    float predict(const std::string& fen) {
        chess::Position pos;
        pos.set(fen);
        std::vector<binpack::TrainingDataEntry> ds{{pos}};

        feed(ds);

        return prediction().to_cpu()(0) * EVAL_SCALE;
    }

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

    void quantize_params(const std::string& path = "examples/astra/quantized_model.nnue") {
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open())
            error("Failed writing quantized parameters to {}", path);

        write_quantized<int16_t>(f, ft.weight.data().to_cpu(), 255);
        write_quantized<int16_t>(f, ft.bias.data().to_cpu(), 255);
        write_quantized<int8_t>(f, l1.weight.data().to_cpu().transpose(), 64);
        write_quantized<float>(f, l1.bias.data().to_cpu());
        write_quantized<float>(f, l2.weight.data().to_cpu().transpose());
        write_quantized<float>(f, l2.bias.data().to_cpu());
        write_quantized<float>(f, l3.weight.data().to_cpu().transpose());
        write_quantized<float>(f, l3.bias.data().to_cpu());
    }

  private:
    sorei::nn::Tensor<int> stm_indices;
    sorei::nn::Tensor<int> nstm_indices;
    sorei::nn::Tensor<int> bucket_indices;
    sorei::nn::Tensor<float> targets;
};
