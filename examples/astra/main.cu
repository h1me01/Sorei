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
    constexpr float lr = 0.001f;
    constexpr int epochs = 100;
    constexpr int batch_size = 16384;
    constexpr int batches_per_epoch = 6104;
    constexpr int report_rate = 100;
    constexpr int save_rate = 20;

    const std::string checkpoint_dir = "examples/astra/checkpoints";

    AstraModel model;
    for (auto p : model.params())
        p->set_bounds(-0.99f, 0.99f);

    auto optim = sorei::nn::optim::AdamW(model.params(), 0.9f, 0.999f, 0.01f);
    auto lr_sched = sorei::nn::lr_sched::CosineAnnealing(lr, lr * std::pow(0.3f, 3), epochs);

    auto binpack_loader = BinpackLoader(
        batch_size,
        2,
        {"/home/h1me/Downloads/data.binpack"},
        [](const binpack::TrainingDataEntry& e) {
            return std::abs(e.score) > 10000 //
                   || e.isInCheck()          //
                   || e.isCapturingMove()    //
                   || e.move.type != chess::MoveType::Normal;
        }
    );

    println("\nTraining configuration:");
    println("  Device         {}", device_info());
    println("  Epochs         {}", epochs);
    println("  Batch Size     {}", batch_size);
    println("  Batches/Epoch  {}", batches_per_epoch);
    println("  Save Rate      {}", save_rate);
    println("  LR Scheduler   {}", lr_sched.info());
    println("  Output Path    {}", checkpoint_dir);

    println("\nTraining Data");
    for (const auto& f : binpack_loader.filenames())
        println("  {}", f);
    println("");

    for (int epoch = 1; epoch <= epochs; epoch++) {
        Timer timer;

        model.clear_running_loss();

        for (int batch = 1; batch <= batches_per_epoch; batch++) {
            model.feed(binpack_loader.next());
            model.backward();
            optim.step(lr_sched.get());

            bool last_batch = (batch == batches_per_epoch);
            if (batch % report_rate == 0 || last_batch) {
                float loss = model.running_loss();
                float time_sec = timer.elapsed() / 1000.0f;

                print(
                    "\repoch/batch = {:3d}/{:4d} | loss = {:1.6f} | pos/sec = {:7d} | time = "
                    "{:3d}s",
                    epoch,
                    batch,
                    loss / batch,
                    (int)std::round((batch_size * batch) / time_sec),
                    (int)std::round(time_sec)
                );

                if (!last_batch)
                    std::cout << std::flush;
                else
                    std::cout << std::endl;
            }
        }

        if (epoch % save_rate == 0) {
            const std::string epoch_checkpoint_dir =
                checkpoint_dir + "/epoch_" + std::to_string(epoch);
            model.save_params(epoch_checkpoint_dir + "/model.nn");
            optim.save_state(epoch_checkpoint_dir + "/optimizer");
        }

        lr_sched.step();
    }

    model.quantize_params();

    println("Prediction: {}", model.predict(chess::Position::startPosition().fen()));
}
