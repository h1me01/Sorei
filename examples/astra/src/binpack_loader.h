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
#include "model.h"
#include "sorei/nn.h"

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
          stream_(std::make_unique<binpack::CompressedTrainingDataEntryParallelReader>(
              num_reader_threads(thread_count_),
              filenames_,
              std::ios::in | std::ios::binary,
              true,
              std::move(skip_predicate)
          )) {

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

                auto batch = new AstraInputs(entries);

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

    AstraInputs* next() {
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
    std::deque<AstraInputs*> batches_;
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
