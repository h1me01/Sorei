#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "sf_binpack/training_data_format.h"
#include "sorei/nn.h"

class BinpackLoader {
  public:
    static constexpr double worker_thread_ratio = 0.14;

    BinpackLoader(
        int batch_size,
        int thread_count,
        std::vector<std::string> filenames,
        std::function<bool(const binpack::TrainingDataEntry&)> skip_predicate = nullptr
    )
        : batch_size_(batch_size),
          thread_count_(thread_count),
          filenames_(std::move(filenames)) {

        validate_files(filenames_);

        const int num_reader_threads = calc_num_reader_threads(thread_count);
        stream_ = std::make_unique<binpack::CompressedTrainingDataEntryParallelReader>(
            num_reader_threads, filenames_, std::ios::in | std::ios::binary, true, skip_predicate
        );

        stop_flag_.store(false);
        num_workers_.store(0);

        const int num_worker_threads = calc_num_worker_threads(thread_count_);
        for (int i = 0; i < num_worker_threads; ++i) {
            workers_.emplace_back(&BinpackLoader::worker_loop, this);
            num_workers_.fetch_add(1);
        }
    }

    BinpackLoader(const BinpackLoader&) = delete;
    BinpackLoader& operator=(const BinpackLoader&) = delete;
    BinpackLoader(BinpackLoader&&) = delete;
    BinpackLoader& operator=(BinpackLoader&&) = delete;

    ~BinpackLoader() {
        stop_flag_.store(true);
        batches_not_full_.notify_all();

        for (auto& worker : workers_)
            if (worker.joinable())
                worker.join();
    }

    std::vector<binpack::TrainingDataEntry> next() {
        std::unique_lock lock(batch_mutex_);
        batches_any_.wait(lock, [this]() { return !batches_.empty() || num_workers_.load() == 0; });

        if (!batches_.empty()) {
            auto batch = std::move(batches_.front());
            batches_.pop_front();

            lock.unlock();
            batches_not_full_.notify_one();

            return batch;
        }

        return {};
    }

    std::vector<std::string> filenames() const { return filenames_; }

  private:
    static int calc_num_worker_threads(int concurrency) {
        return std::max(1, static_cast<int>(std::floor(concurrency * worker_thread_ratio)));
    }

    static int calc_num_reader_threads(int concurrency) {
        return std::max(1, concurrency - calc_num_worker_threads(concurrency));
    }

    void worker_loop() {
        std::vector<binpack::TrainingDataEntry> entries;
        entries.reserve(batch_size_);

        while (!stop_flag_.load()) {
            entries.clear();

            stream_->fill(entries, batch_size_);
            if (entries.empty())
                break;

            {
                std::unique_lock lock(batch_mutex_);
                batches_not_full_.wait(lock, [this]() {
                    return batches_.size() < static_cast<std::size_t>(thread_count_ + 1) ||
                           stop_flag_.load();
                });

                batches_.emplace_back(std::move(entries));

                lock.unlock();
                batches_any_.notify_one();
            }
        }

        num_workers_.fetch_sub(1);
        batches_any_.notify_one();
    }

  private:
    int batch_size_;
    int thread_count_;
    std::vector<std::string> filenames_;

    std::deque<std::vector<binpack::TrainingDataEntry>> batches_;

    std::mutex batch_mutex_;

    std::condition_variable batches_not_full_;
    std::condition_variable batches_any_;

    std::atomic_bool stop_flag_;
    std::atomic_int num_workers_;

    std::vector<std::thread> workers_;
    std::unique_ptr<binpack::CompressedTrainingDataEntryParallelReader> stream_;

    static void validate_files(const std::vector<std::string>& files) {
        if (files.empty())
            sorei::error("BinpackLoader: no training data files provided");

        for (const auto& f : files) {
            if (!std::filesystem::exists(f))
                sorei::error("BinpackLoader: missing file {}", f);

            std::ifstream file(f);
            if (!file.is_open())
                sorei::error("BinpackLoader: cannot open file (permission or invalid path): {}", f);
        }

        for (const auto& f : files)
            if (!f.ends_with(".binpack"))
                sorei::error("BinpackLoader: {} is not a binpack file", f);
    }
};
