#include <chrono>
#include <cmath>
#include <iostream>

#include "dataloader.h"
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

static float eval_accuracy(MNISTModel& model, const MNISTDataset& dataset, int batch_size) {
    MNISTLoader loader(dataset, batch_size, /*seed=*/0);
    const int num_batches = dataset.size() / batch_size;
    int correct = 0;

    for (int b = 0; b < num_batches; b++) {
        auto [images, labels] = loader.next();
        model.feed(images, labels);

        auto cpu_logits = model.prediction().to_cpu();
        for (int s = 0; s < batch_size; s++) {
            int pred = 0;
            for (int c = 1; c < MNISTModel::NUM_CLASSES; c++)
                if (cpu_logits(c, s) > cpu_logits(pred, s))
                    pred = c;
            if (pred == labels[s])
                correct++;
        }
    }

    return 100.0f * correct / (num_batches * batch_size);
}

int main() {
    constexpr float lr = 0.001f;
    constexpr int epochs = 20;
    constexpr int batch_size = 256;

    MNISTDataset train_set("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
    MNISTDataset test_set("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");

    println("{} training samples", train_set.size());
    println("{} test samples\n", test_set.size());

    MNISTLoader train_loader(train_set, batch_size);
    const int batches_per_epoch = train_loader.batches_per_epoch();

    MNISTModel model;
    auto optim = sorei::nn::optim::AdamW(model.params(), 0.9f, 0.999f, 0.01f);
    auto lr_sched = sorei::nn::lr_sched::CosineAnnealing(lr, lr * 0.1f, epochs);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        Timer timer;
        model.clear_running_loss();

        for (int batch = 1; batch <= batches_per_epoch; batch++) {
            auto [images, labels] = train_loader.next();
            model.feed(images, labels);
            model.backward();
            optim.step(lr_sched.get());
        }

        print(
            "\repoch {:2d}/{} | loss {:.5f} | test accuracy: {:.2f}%\n",
            epoch,
            epochs,
            model.running_loss() / batches_per_epoch,
            eval_accuracy(model, test_set, batch_size)
        );

        lr_sched.step();
    }

    return 0;
}
