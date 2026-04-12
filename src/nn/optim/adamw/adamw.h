#pragma once

#include <filesystem>
#include <fstream>

#include "../optimizer.h"

namespace sorei::nn::optim {

class AdamW : public Optimizer {
  public:
    AdamW(
        std::vector<layer::Param*> params, float beta1 = 0.9, float beta2 = 0.999, float decay = 0.0
    )
        : Optimizer(std::move(params)),
          beta1_(beta1),
          beta2_(beta2),
          decay_(decay) {

        SOREI_CHECK(beta1 >= 0.0f && beta1 < 1.0f);
        SOREI_CHECK(beta2 >= 0.0f && beta2 < 1.0f);

        for (auto* t : params_) {
            int size = t->data().size();
            momentum_.emplace_back(size);
            velocity_.emplace_back(size);
        }
    }

    void load_state(const std::string& path) override {
        const std::filesystem::path state_path = std::filesystem::path(path);
        if (!std::filesystem::exists(state_path))
            error("Optimizer: state directory does not exist {}", state_path.string());

        try {
            load_buffers(state_path / "momentum.bin", momentum_);
            load_buffers(state_path / "velocity.bin", velocity_);
        } catch (const std::exception& e) {
            error(
                "Optimizer: failed to load state from {}, error {}", state_path.string(), e.what()
            );
        }
    }

    void save_state(const std::string& path) const override {
        const std::filesystem::path state_path = std::filesystem::path(path);

        try {
            std::filesystem::create_directories(state_path);

            save_buffers(state_path / "momentum.bin", momentum_);
            save_buffers(state_path / "velocity.bin", velocity_);
        } catch (const std::exception& e) {
            error("Optimizer: failed to save state to {}, error {}", state_path.string(), e.what());
        }
    }

    void step(float lr) override;

  private:
    float beta1_;
    float beta2_;
    float decay_;

    std::vector<tensor::GPUArray<float>> momentum_;
    std::vector<tensor::GPUArray<float>> velocity_;

    void save_buffers(
        const std::string& file, const std::vector<tensor::GPUArray<float>>& buffers
    ) const {
        std::ofstream f(file, std::ios::binary);
        if (!f.is_open())
            error("Optimizer: failed to open state file for writing {}", file);

        for (const auto& buf : buffers) {
            const auto host_data = buf.to_cpu();

            const size_t element_count = static_cast<size_t>(host_data.size());
            const size_t bytes_to_write = element_count * sizeof(float);

            f.write(reinterpret_cast<const char*>(host_data.data()), bytes_to_write);

            if (!f.good()) {
                error(
                    "Optimizer: failed writing state to file {}, expected to write {} elements",
                    file,
                    element_count
                );
            }
        }
    }

    void load_buffers(const std::string& file, std::vector<tensor::GPUArray<float>>& buffers) {
        std::ifstream f(file, std::ios::binary);
        if (!f.is_open())
            error("Optimizer: failed to open state file {}", file);

        for (auto& buf : buffers) {
            tensor::CPUArray<float> cpu_buffer(buf.size());

            const size_t element_count = static_cast<size_t>(cpu_buffer.size());
            const size_t bytes_to_read = element_count * sizeof(float);

            f.read(reinterpret_cast<char*>(cpu_buffer.data()), bytes_to_read);

            const size_t bytes_read = static_cast<size_t>(f.gcount());
            if (bytes_read != bytes_to_read) {
                error(
                    "Optimizer: failed to read state from file {}, expected to read {} elements "
                    "but "
                    "read "
                    "{}",
                    file,
                    element_count,
                    bytes_read / sizeof(float)
                );
            }

            buf.upload(cpu_buffer);
        }
    }
};

} // namespace sorei::nn::optim
