#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "sorei/nn.h"

class MNISTDataset {
  public:
    MNISTDataset(const std::string& images_path, const std::string& labels_path) {
        std::ifstream fi(images_path, std::ios::binary);
        if (!fi) {
            std::cout << "MNIST: cannot open images file '" << images_path << "'\n";
            std::abort();
        }

        SOREI_CHECK(read_be32(fi) == 0x00000803u);
        n_ = read_be32(fi);
        rows_ = read_be32(fi);
        cols_ = read_be32(fi);
        dim_ = rows_ * cols_;

        std::vector<uint8_t> raw(n_ * dim_);
        fi.read(reinterpret_cast<char*>(raw.data()), raw.size());
        SOREI_CHECK(fi);

        pixels_.resize(n_ * dim_);
        for (int i = 0; i < (int)raw.size(); i++)
            pixels_[i] = raw[i] / 255.0f;

        std::ifstream fl(labels_path, std::ios::binary);
        if (!fl) {
            std::cout << "MNIST: cannot open labels file '" << labels_path << "'\n";
            std::abort();
        }

        SOREI_CHECK(read_be32(fl) == 0x00000801u);
        SOREI_CHECK(read_be32(fl) == (uint32_t)n_);

        std::vector<uint8_t> raw_lbl(n_);
        fl.read(reinterpret_cast<char*>(raw_lbl.data()), n_);
        SOREI_CHECK(fl);

        labels_.resize(n_);
        for (int i = 0; i < n_; i++)
            labels_[i] = raw_lbl[i];
    }

    int size() const { return n_; }
    int dim() const { return dim_; }
    float pixel(int sample, int px) const { return pixels_[sample * dim_ + px]; }
    int label(int sample) const { return labels_[sample]; }

  private:
    int n_, rows_, cols_, dim_;
    std::vector<float> pixels_;
    std::vector<int> labels_;

    static uint32_t read_be32(std::ifstream& f) {
        uint8_t b[4];
        f.read(reinterpret_cast<char*>(b), 4);
        return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | b[3];
    }
};

class MNISTLoader {
  public:
    MNISTLoader(const MNISTDataset& dataset, int batch_size, uint32_t seed = 42)
        : dataset_(dataset),
          batch_size_(batch_size),
          perm_(dataset.size()),
          images_buf_({dataset.dim(), batch_size}),
          labels_buf_({batch_size}),
          rng_(seed) {

        std::iota(perm_.begin(), perm_.end(), 0);
        shuffle();
    }

    struct Batch {
        const sorei::nn::Tensor<float>& images;
        const sorei::nn::Tensor<int>& labels;
    };

    Batch next() {
        if (pos_ + batch_size_ > (int)perm_.size())
            shuffle();

        const int dim = dataset_.dim();
        for (int b = 0; b < batch_size_; b++) {
            const int idx = perm_[pos_ + b];
            for (int p = 0; p < dim; p++)
                images_buf_(p, b) = dataset_.pixel(idx, p);
            labels_buf_[b] = dataset_.label(idx);
        }
        pos_ += batch_size_;

        return {images_buf_, labels_buf_};
    }

    int batches_per_epoch() const { return dataset_.size() / batch_size_; }
    void reset() { shuffle(); }

  private:
    const MNISTDataset& dataset_;
    int batch_size_;
    int pos_ = 0;
    std::vector<int> perm_;
    sorei::nn::Tensor<float> images_buf_;
    sorei::nn::Tensor<int> labels_buf_;
    std::mt19937 rng_;

    void shuffle() {
        std::shuffle(perm_.begin(), perm_.end(), rng_);
        pos_ = 0;
    }
};
