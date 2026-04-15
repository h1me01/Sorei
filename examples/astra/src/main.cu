#include <chrono>
#include <filesystem>

#include "binpack_loader.h"
#include "model.h"

class Timer {
    using Clock = std::chrono::steady_clock;

  public:
    Timer() { start_ = Clock::now(); }

    template <typename Duration = std::chrono::milliseconds>
    long long elapsed() const {
        return std::chrono::duration_cast<Duration>(Clock::now() - start_).count();
    }

  private:
    Clock::time_point start_;
};

int main() {
    const float lr = 0.001f;
    const int epochs = 400;
    const int batch_size = 16384;
    const int batches_per_epoch = 6104;
    const int save_rate = 40;

    const std::string checkpoint_dir = "checkpoints";

    AstraModel model;
    for (auto p : model.params())
        p->set_bounds(-0.99f, 0.99f);

    auto optim = sorei::nn::optim::AdamW(model.params(), 0.9f, 0.999f, 0.01f);
    auto lr_sched = sorei::nn::lr_sched::CosineAnnealingLR(lr, lr * std::pow(0.3f, 3), epochs);

    auto binpack_loader = BinpackLoader(
        batch_size,
        4,
        {"/home/h1me/Downloads/data.binpack"},
        [](const binpack::TrainingDataEntry& e) {
            return std::abs(e.score) > 10000 //
                   || e.isInCheck()          //
                   || e.isCapturingMove()    //
                   || e.move.type != chess::MoveType::Normal;
        }
    );

    auto prefetcher = BatchPrefetcher(binpack_loader, batch_size);

    sorei::println("\nTraining configuration:");
    sorei::println("  Device         {}", sorei::device_info());
    sorei::println("  Epochs         {}", epochs);
    sorei::println("  Batch Size     {}", batch_size);
    sorei::println("  Batches/Epoch  {}", batches_per_epoch);
    sorei::println("  Save Rate      {}", save_rate);
    sorei::println("  LR Scheduler   {}", lr_sched.info());
    sorei::println("  Output Path    {}", checkpoint_dir);

    sorei::println("\nTraining Data");
    for (const auto& f : binpack_loader.filenames())
        sorei::println("  {}", f);
    sorei::println("");

    for (int epoch = 1; epoch <= epochs; epoch++) {
        Timer timer;

        model.clear_running_loss();

        for (int batch = 1; batch <= batches_per_epoch; batch++) {
            auto* dev_batch = prefetcher.next();
            if (!dev_batch)
                break;

            model.forward(
                {{"stm_in", dev_batch->stm_indices},
                 {"nstm_in", dev_batch->nstm_indices},
                 {"output_bucket", dev_batch->bucket_indices},
                 {"target", dev_batch->targets}}
            );
            model.backward();
            optim.step(lr_sched.get_lr());

            if (batch % 100 == 0 || batch == batches_per_epoch) {
                float time_sec = timer.elapsed() / 1000.0f;
                sorei::print(
                    "\repoch/batch = {:3d}/{:4d} | loss = {:1.6f} | pos/sec = {:7d} | time = "
                    "{:3d}s",
                    epoch,
                    batch,
                    model.running_loss() / batch,
                    (int)std::round((batch_size * batch) / time_sec),
                    (int)std::round(time_sec)
                );

                if (batch != batches_per_epoch)
                    std::cout << std::flush;
                else
                    sorei::println("");
            }
        }

        if (epoch % save_rate == 0) {
            std::string e_checkpoint_dir = checkpoint_dir + "/epoch_" + std::to_string(epoch);
            std::filesystem::create_directories(e_checkpoint_dir);
            model.save_params(e_checkpoint_dir + "/model.nn");
            optim.save_state(e_checkpoint_dir + "/optimizer");
        }

        lr_sched.step();
    }

    model.quantize_params();

    sorei::println(
        "\nstartpos eval: {:.2f}", model.predict(chess::Position::startPosition().fen())
    );
}
