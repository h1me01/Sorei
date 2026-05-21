#pragma once

#include <array>
#include <fstream>
#include <ranges>

#include "../sf_binpack/training_data_format.h"
#include "sorei/nn.h"

class AstraInputs {
  public:
    static constexpr std::array<int, 64> INPUT_BUCKET = {
        0, 1, 2, 3, 3, 2, 1, 0, //
        4, 5, 6, 7, 7, 6, 5, 4, //
        8, 8, 8, 8, 8, 8, 8, 8, //
        9, 9, 9, 9, 9, 9, 9, 9, //
        9, 9, 9, 9, 9, 9, 9, 9, //
        9, 9, 9, 9, 9, 9, 9, 9, //
        9, 9, 9, 9, 9, 9, 9, 9, //
        9, 9, 9, 9, 9, 9, 9, 9, //
    };

    static constexpr int NUM_INPUT_BUCKETS = std::ranges::max(INPUT_BUCKET) + 1;

    static constexpr int FEATURE_SIZE = 768;
    static constexpr float EVAL_SCALE = 400.0f;

    sorei::matrix::HostPinnedMatrix<int> stm_indices;
    sorei::matrix::HostPinnedMatrix<int> nstm_indices;
    sorei::matrix::HostPinnedMatrix<int> bucket_indices;
    sorei::matrix::HostPinnedMatrix<float> targets;

    void
    fill(const std::vector<binpack::TrainingDataEntry>& entries, const float wdl_weight = 0.5f) {
        const int n = static_cast<int>(entries.size());

        stm_indices.resize({32, n});
        nstm_indices.resize({32, n});
        bucket_indices.resize({1, n});
        targets.resize({1, n});

        stm_indices.fill(-1);
        nstm_indices.fill(-1);

        for (int i = 0; i < n; ++i) {
            const auto& pos = entries[i].pos;
            const auto stm = pos.sideToMove();
            const auto stm_ksq = pos.kingSquare(stm);
            const auto nstm_ksq = pos.kingSquare(!stm);

            int j = 0;
            for (auto sq : pos.piecesBB()) {
                const auto pc = pos.pieceAt(sq);
                stm_indices(j, i) = feature_index(pc, sq, stm_ksq, stm);
                nstm_indices(j, i) = feature_index(pc, sq, nstm_ksq, !stm);
                ++j;
            }

            const float score_target = sigmoid(entries[i].score / EVAL_SCALE);
            const float wdl_target = (entries[i].result + 1) / 2.0f;
            targets(i) = wdl_weight * wdl_target + (1.0f - wdl_weight) * score_target;
            bucket_indices(i) = (pos.pieceCount() - 2) / 4;
        }
    }

  private:
    static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

    static int
    feature_index(chess::Piece pc, chess::Square sq, chess::Square ksq, chess::Color view) {
        if (view == chess::Color::Black) {
            sq.flipVertically();
            ksq.flipVertically();
        }

        if (ksq.file() > chess::fileD)
            sq.flipHorizontally();

        return (int)sq                         //
               + (int)pc.type() * 64           //
               + (pc.color() != view) * 64 * 6 //
               + INPUT_BUCKET[(int)ksq] * FEATURE_SIZE;
    }
};

struct AstraModel : sorei::nn::Model {
    static constexpr int FT_SIZE = 1536;
    static constexpr int L1_SIZE = 16;
    static constexpr int L2_SIZE = 32;
    static constexpr int OUTPUT_BUCKETS = 8;

    sorei::nn::ParamNode factorizer;
    sorei::nn::AffineLayer ft, l1, l2, l3;

    sorei::nn::GraphOutput build_graph(sorei::nn::GraphBuilder& b) override {
        // params
        factorizer = b.param(AstraInputs::FEATURE_SIZE, FT_SIZE);
        ft = b.affine_layer(AstraInputs::NUM_INPUT_BUCKETS * AstraInputs::FEATURE_SIZE, FT_SIZE);
        l1 = b.affine_layer(FT_SIZE, L1_SIZE * OUTPUT_BUCKETS);
        l2 = b.affine_layer(2 * L1_SIZE, L2_SIZE * OUTPUT_BUCKETS);
        l3 = b.affine_layer(L2_SIZE, OUTPUT_BUCKETS);

        // inputs
        auto stm_in = b.input_int("stm_in", {32, 0});
        auto nstm_in = b.input_int("nstm_in", {32, 0});
        auto output_bucket = b.bucket_index("output_bucket", OUTPUT_BUCKETS, 0);
        auto target = b.input_float("target", {1, 0});

        // forward pass
        auto repeated_factorizer = b.concat(
            std::vector<sorei::nn::Node>(AstraInputs::NUM_INPUT_BUCKETS, factorizer),
            sorei::nn::ConcatAxis::Cols
        );
        auto factorized_ftw = repeated_factorizer + ft.weight;

        auto ft_stm = b.affine(stm_in, factorized_ftw, ft.bias).clamped_relu().pairwise_mul();
        auto ft_nstm = b.affine(nstm_in, factorized_ftw, ft.bias).clamped_relu().pairwise_mul();
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

        AstraInputs batch;
        batch.fill(ds);

        forward({
            {"stm_in", batch.stm_indices},
            {"nstm_in", batch.nstm_indices},
            {"output_bucket", batch.bucket_indices},
            {"target", batch.targets},
        });

        return prediction().to_host()(0) * AstraInputs::EVAL_SCALE;
    }

    void quantize_params(const std::string& path = "quantized_model.nnue") {
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open())
            sorei::error("Failed writing quantized parameters to {}", path);

        auto facto = factorizer.data().to_host();
        auto ftw = ft.weight.data().to_host();

        sorei::matrix::HostMatrix<float> factorized_ftw(ftw.shape());
        for (int r = 0; r < ftw.rows(); r++)
            for (int c = 0; c < ftw.cols(); c++)
                factorized_ftw(r, c) = ftw(r, c) + facto(r, c % AstraInputs::FEATURE_SIZE);

        write_quantized<int16_t>(f, factorized_ftw, 255);
        write_quantized<int16_t>(f, ft.bias.data().to_host(), 255);
        write_quantized<int8_t>(f, l1.weight.data().to_host().transpose(), 64);
        write_quantized(f, l1.bias.data().to_host());
        write_quantized(f, l2.weight.data().to_host().transpose());
        write_quantized(f, l2.bias.data().to_host());
        write_quantized(f, l3.weight.data().to_host().transpose());
        write_quantized(f, l3.bias.data().to_host());
    }

  private:
    template <typename T = float>
    void
    write_quantized(std::ostream& f, const sorei::matrix::HostMatrix<float>& src, int scale = 1) {
        sorei::matrix::HostMatrix<T> dst(src.shape());

        for (int i = 0; i < src.size(); i++) {
            if constexpr (std::is_same_v<T, float>) {
                dst(i) = src(i);
            } else {
                constexpr double lo = std::numeric_limits<T>::min();
                constexpr double hi = std::numeric_limits<T>::max();
                const double qv = std::round((double)scale * (double)src(i));
                if (qv < lo || qv > hi)
                    sorei::println("Warning: value {} out of range, clamping", src(i));
                dst(i) = std::clamp(qv, lo, hi);
            }
        }

        f.write(reinterpret_cast<const char*>(dst.data()), dst.size() * sizeof(T));
        if (!f.good())
            sorei::error("Failed writing quantized data");
    }
};
