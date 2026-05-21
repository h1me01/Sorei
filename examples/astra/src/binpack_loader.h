#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../sf_binpack/training_data_format.h"
#include "model.h"
#include "sorei/nn.h"

class XorShift64 {
  public:
    XorShift64() {
        auto ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        state_ = static_cast<uint64_t>(ns) | 1ULL;
    }

    uint64_t next() {
        state_ ^= state_ << 13;
        state_ ^= state_ >> 7;
        state_ ^= state_ << 17;
        return state_;
    }

  private:
    uint64_t state_;
};

struct CompactEntry {
    chess::CompressedPosition pos;
    std::int16_t score;
    std::int16_t result;
};
static_assert(sizeof(CompactEntry) == 32);

class BinpackLoader {
  public:
    static constexpr double WORKER_THREAD_RATIO = 0.14;

    using SkipPredicate = std::function<bool(const binpack::TrainingDataEntry&)>;

    BinpackLoader(
        int batch_size,
        int thread_count,
        std::vector<std::string> filenames,
        size_t shuffle_buffer_size = 0,
        SkipPredicate skip_predicate = nullptr
    )
        : batch_size_(batch_size),
          thread_count_(thread_count),
          filenames_(std::move(filenames)),
          shuffle_buffer_size_(shuffle_buffer_size),
          num_workers_(shuffle_buffer_size > 0 ? 1 : num_worker_threads(thread_count)),
          stream_(std::make_unique<binpack::CompressedTrainingDataEntryParallelReader>(
              num_reader_threads(thread_count_),
              filenames_,
              std::ios::in | std::ios::binary,
              true,
              std::move(skip_predicate)
          )) {

        validate_files(filenames_);
        stop_flag_.store(false);

        if (shuffle_buffer_size_ > 0) {
            workers_.emplace_back([this]() { run_shuffle_producer(); });
        } else {
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
    }

    BinpackLoader(const BinpackLoader&) = delete;
    BinpackLoader& operator=(const BinpackLoader&) = delete;
    BinpackLoader(BinpackLoader&&) = delete;
    BinpackLoader& operator=(BinpackLoader&&) = delete;

    ~BinpackLoader() {
        stop_flag_.store(true);
        batches_not_full_.notify_all();
        batches_any_.notify_all();
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
    size_t shuffle_buffer_size_;
    std::deque<AstraInputs*> batches_;
    std::mutex batch_mutex_;
    std::condition_variable batches_not_full_;
    std::condition_variable batches_any_;
    std::atomic_bool stop_flag_;
    std::atomic_int num_workers_;
    std::vector<std::thread> workers_;
    std::unique_ptr<binpack::CompressedTrainingDataEntryParallelReader> stream_;

    void fill_shuffle_buf(std::vector<CompactEntry>& buf) {
        buf.clear();
        while (buf.size() < shuffle_buffer_size_ && !stop_flag_.load()) {
            const int want =
                static_cast<int>(std::min<size_t>(16384, shuffle_buffer_size_ - buf.size()));
            std::vector<binpack::TrainingDataEntry> tmp;
            stream_->fill(tmp, want);
            if (tmp.empty())
                break;
            for (auto& e : tmp)
                buf.push_back({e.pos.compress(), e.score, e.result});
        }
    }

    void run_shuffle_producer() {
        XorShift64 rng;

        std::vector<CompactEntry> cur_buf, nxt_buf;
        cur_buf.reserve(shuffle_buffer_size_);
        nxt_buf.reserve(shuffle_buffer_size_);

        fill_shuffle_buf(cur_buf);
        if (cur_buf.empty()) {
            num_workers_.fetch_sub(1);
            batches_any_.notify_one();
            return;
        }

        while (!stop_flag_.load()) {
            auto fill_future =
                std::async(std::launch::async, [this, &nxt_buf]() { fill_shuffle_buf(nxt_buf); });

            for (int64_t i = static_cast<int64_t>(cur_buf.size()) - 1; i > 0; --i) {
                const int64_t j = static_cast<int64_t>(rng.next() % static_cast<uint64_t>(i + 1));
                std::swap(cur_buf[i], cur_buf[j]);
            }

            for (size_t pos = 0;
                 pos + static_cast<size_t>(batch_size_) <= cur_buf.size() && !stop_flag_.load();
                 pos += static_cast<size_t>(batch_size_)) {

                std::vector<binpack::TrainingDataEntry> entries;
                entries.reserve(static_cast<size_t>(batch_size_));
                for (int i = 0; i < batch_size_; ++i) {
                    const CompactEntry& ce = cur_buf[pos + static_cast<size_t>(i)];
                    binpack::TrainingDataEntry e;
                    e.pos = ce.pos.decompress();
                    e.score = ce.score;
                    e.result = ce.result;
                    entries.push_back(std::move(e));
                }
                auto* batch = new AstraInputs(entries);

                {
                    std::unique_lock lock(batch_mutex_);
                    batches_not_full_.wait(lock, [this]() {
                        return batches_.size() < static_cast<size_t>(thread_count_ + 1) ||
                               stop_flag_.load();
                    });
                    batches_.emplace_back(batch);
                    lock.unlock();
                    batches_any_.notify_one();
                }
            }

            fill_future.wait();
            if (nxt_buf.empty())
                break;

            std::swap(cur_buf, nxt_buf);
        }

        num_workers_.fetch_sub(1);
        batches_any_.notify_one();
    }

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
