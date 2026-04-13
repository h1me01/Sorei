#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../sf_binpack/training_data_format.h"
#include "sorei/nn.h"

// BatchData

class BatchData {
  public:
    static constexpr float EVAL_SCALE = 400.0f;
    static constexpr float WDL_WEIGHT = 0.7f;

    explicit BatchData(const std::vector<binpack::TrainingDataEntry>& entries) {
        const int n = static_cast<int>(entries.size());

        stm_indices_.resize({32, n});
        nstm_indices_.resize({32, n});
        bucket_indices_.resize({n});
        targets_.resize({n});

        stm_indices_.fill(-1);
        nstm_indices_.fill(-1);

        for (int i = 0; i < n; ++i) {
            const auto& e = entries[i];
            const auto stm = e.pos.sideToMove();

            int j = 0;
            for (auto sq : e.pos.piecesBB()) {
                const auto pc = e.pos.pieceAt(sq);
                stm_indices_(j, i) = feature_index(pc, sq, stm);
                nstm_indices_(j, i) = feature_index(pc, sq, !stm);
                ++j;
            }

            const float score_target = sigmoid(e.score / EVAL_SCALE);
            const float wdl_target = (e.result + 1) / 2.0f;
            targets_[i] = WDL_WEIGHT * wdl_target + (1.0f - WDL_WEIGHT) * score_target;
            bucket_indices_[i] = (e.pos.pieceCount() - 2) / 4;
        }
    }

    const sorei::nn::Tensor<int>& stm_indices() const { return stm_indices_; }
    const sorei::nn::Tensor<int>& nstm_indices() const { return nstm_indices_; }
    const sorei::nn::Tensor<int>& bucket_indices() const { return bucket_indices_; }
    const sorei::nn::Tensor<float>& targets() const { return targets_; }

  private:
    sorei::nn::Tensor<int> stm_indices_;
    sorei::nn::Tensor<int> nstm_indices_;
    sorei::nn::Tensor<int> bucket_indices_;
    sorei::nn::Tensor<float> targets_;

    static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

    static int feature_index(chess::Piece pc, chess::Square sq, chess::Color view) {
        if (view == chess::Color::Black)
            sq.flipVertically();
        return static_cast<int>(sq) + static_cast<int>(pc.type()) * 64 +
               (pc.color() != view) * 64 * 6;
    }
};

// BinpackLoader

class BinpackLoader {
  public:
    static constexpr double WORKER_THREAD_RATIO = 0.14;

    using SkipPredicate = std::function<bool(const binpack::TrainingDataEntry&)>;

    BinpackLoader(
        int batch_size,
        int thread_count,
        std::vector<std::string> filenames,
        SkipPredicate skip_predicate = nullptr
    )
        : batch_size_(batch_size),
          thread_count_(thread_count),
          filenames_(std::move(filenames)),
          num_workers_(num_worker_threads(thread_count)),
          stream_(
              std::make_unique<binpack::CompressedTrainingDataEntryParallelReader>(
                  num_reader_threads(thread_count_),
                  filenames_,
                  std::ios::in | std::ios::binary,
                  true,
                  std::move(skip_predicate)
              )
          ) {

        validate_files(filenames_);

        stop_flag_.store(false);

        auto worker = [this]() {
            std::vector<binpack::TrainingDataEntry> entries;
            entries.reserve(batch_size_);

            while (!stop_flag_.load()) {
                entries.clear();
                {
                    stream_->fill(entries, batch_size_);
                    if (entries.empty())
                        break;
                }

                auto batch = new BatchData(entries);

                {
                    std::unique_lock lock(batch_mutex_);
                    batches_not_full_.wait(lock, [this]() {
                        return batches_.size() < thread_count_ + 1 || stop_flag_.load();
                    });
                    batches_.emplace_back(batch);
                    lock.unlock();
                    batches_any_.notify_one();
                }
            }
            num_workers_.fetch_sub(1);
            batches_any_.notify_one();
        };

        for (int i = 0; i < num_worker_threads(thread_count_); ++i)
            workers_.emplace_back(worker);
    }

    BinpackLoader(const BinpackLoader&) = delete;
    BinpackLoader& operator=(const BinpackLoader&) = delete;
    BinpackLoader(BinpackLoader&&) = delete;
    BinpackLoader& operator=(BinpackLoader&&) = delete;

    ~BinpackLoader() {
        stop_flag_.store(true);
        batches_not_full_.notify_all();
        for (auto& w : workers_)
            if (w.joinable())
                w.join();
        for (auto* b : batches_)
            delete b;
    }

    BatchData* next() {
        std::unique_lock lock(batch_mutex_);
        batches_any_.wait(lock, [this] { return !batches_.empty() || num_workers_.load() == 0; });
        if (batches_.empty())
            return nullptr;

        auto* batch = batches_.front();
        batches_.pop_front();
        lock.unlock();
        batches_not_full_.notify_one();
        return batch;
    }

    const std::vector<std::string>& filenames() const { return filenames_; }

  private:
    int batch_size_;
    int thread_count_;
    std::vector<std::string> filenames_;
    std::deque<BatchData*> batches_;
    std::mutex batch_mutex_;
    std::condition_variable batches_not_full_;
    std::condition_variable batches_any_;
    std::atomic_bool stop_flag_;
    std::atomic_int num_workers_;
    std::vector<std::thread> workers_;
    std::unique_ptr<binpack::CompressedTrainingDataEntryParallelReader> stream_;

    static int num_worker_threads(int concurrency) {
        return std::max(1, static_cast<int>(std::floor(concurrency * WORKER_THREAD_RATIO)));
    }

    static int num_reader_threads(int concurrency) {
        return std::max(1, concurrency - num_worker_threads(concurrency));
    }

    static void validate_files(const std::vector<std::string>& files) {
        if (files.empty())
            sorei::error("BinpackLoader: no training data files provided");
        for (const auto& f : files) {
            if (!std::filesystem::exists(f))
                sorei::error("BinpackLoader: missing file {}", f);
            if (!std::ifstream(f).is_open())
                sorei::error("BinpackLoader: cannot open file: {}", f);
            if (!f.ends_with(".binpack"))
                sorei::error("BinpackLoader: {} is not a binpack file", f);
        }
    }
};

// GPUBatchData

struct GPUBatchData {
    sorei::nn::Tensor<int> stm_indices;
    sorei::nn::Tensor<int> nstm_indices;
    sorei::nn::Tensor<int> bucket_indices;
    sorei::nn::Tensor<float> targets;

    explicit GPUBatchData(int n) {
        using TI = sorei::nn::Tensor<int>;
        using TF = sorei::nn::Tensor<float>;

        stm_indices = TI({32, n}, TI::GPU);
        nstm_indices = TI({32, n}, TI::GPU);
        bucket_indices = TI({n}, TI::GPU);
        targets = TF({n}, TF::GPU);

        stm_staging_ = TI({32, n}, TI::CPU_PINNED);
        nstm_staging_ = TI({32, n}, TI::CPU_PINNED);
        bucket_staging_ = TI({n}, TI::CPU_PINNED);
        targets_staging_ = TF({n}, TF::CPU_PINNED);
    }

    void upload_async(const BatchData& src, cudaStream_t stream) {
        auto stage = [](auto& pinned, const auto& cpu) {
            std::memcpy(pinned.data(), cpu.data(), cpu.bytes());
        };
        stage(stm_staging_.cpu_pinned_data(), src.stm_indices().cpu_data());
        stage(nstm_staging_.cpu_pinned_data(), src.nstm_indices().cpu_data());
        stage(bucket_staging_.cpu_pinned_data(), src.bucket_indices().cpu_data());
        stage(targets_staging_.cpu_pinned_data(), src.targets().cpu_data());

        auto dma = [&](auto& dst, const auto& pinned) {
            SOREI_CUDA_CHECK(cudaMemcpyAsync(
                dst.data(), pinned.data(), pinned.bytes(), cudaMemcpyHostToDevice, stream
            ));
        };
        dma(stm_indices.gpu_data(), stm_staging_.cpu_pinned_data());
        dma(nstm_indices.gpu_data(), nstm_staging_.cpu_pinned_data());
        dma(bucket_indices.gpu_data(), bucket_staging_.cpu_pinned_data());
        dma(targets.gpu_data(), targets_staging_.cpu_pinned_data());
    }

  private:
    sorei::nn::Tensor<int> stm_staging_;
    sorei::nn::Tensor<int> nstm_staging_;
    sorei::nn::Tensor<int> bucket_staging_;
    sorei::nn::Tensor<float> targets_staging_;
};

// BatchPrefetcher

class BatchPrefetcher {
  public:
    BatchPrefetcher(BinpackLoader& loader, int batch_size)
        : loader_(loader) {
        SOREI_CUDA_CHECK(cudaStreamCreate(&stream_));
        SOREI_CUDA_CHECK(cudaEventCreate(&event_));

        buf_[0] = new GPUBatchData(batch_size);
        buf_[1] = new GPUBatchData(batch_size);

        next_cpu_ = loader_.next();
        if (next_cpu_) {
            buf_[write_]->upload_async(*next_cpu_, stream_);
            SOREI_CUDA_CHECK(cudaEventRecord(event_, stream_));
            delete next_cpu_;
            next_cpu_ = loader_.next();
            has_pending_ = true;
        }
    }

    ~BatchPrefetcher() {
        SOREI_CUDA_CHECK(cudaStreamSynchronize(stream_));
        SOREI_CUDA_CHECK(cudaEventDestroy(event_));
        SOREI_CUDA_CHECK(cudaStreamDestroy(stream_));
        delete buf_[0];
        delete buf_[1];
        delete next_cpu_;
    }

    BatchPrefetcher(const BatchPrefetcher&) = delete;
    BatchPrefetcher& operator=(const BatchPrefetcher&) = delete;
    BatchPrefetcher(BatchPrefetcher&&) = delete;
    BatchPrefetcher& operator=(BatchPrefetcher&&) = delete;

    GPUBatchData* next() {
        if (!has_pending_)
            return nullptr;

        SOREI_CUDA_CHECK(cudaEventSynchronize(event_));
        std::swap(read_, write_);
        has_pending_ = false;

        if (next_cpu_) {
            buf_[write_]->upload_async(*next_cpu_, stream_);
            SOREI_CUDA_CHECK(cudaEventRecord(event_, stream_));
            delete next_cpu_;
            next_cpu_ = loader_.next();
            has_pending_ = true;
        }
        return buf_[read_];
    }

  private:
    BinpackLoader& loader_;
    GPUBatchData* buf_[2] = {};
    BatchData* next_cpu_ = nullptr;
    int read_ = 1;
    int write_ = 0;
    bool has_pending_ = false;
    cudaStream_t stream_ = {};
    cudaEvent_t event_ = {};
};
