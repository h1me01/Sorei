#include "softmax_cross_entropy.h"

namespace sorei::nn::layer {

__global__ void softmax_cross_entropy_backward_kernel(
    const float* probs,
    const int* labels,
    const float* out_g,
    float* logits_g,
    const int batch_size,
    const int num_classes
) {
    const int sample = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    if (sample >= batch_size)
        return;

    const float* sample_probs = probs + sample * num_classes;
    float* sample_grad = logits_g + sample * num_classes;

    const int label = labels[sample];
    const float scale = out_g[sample];

    for (int c = tid; c < num_classes; c += num_threads) {
        float g = sample_probs[c];
        if (c == label)
            g -= 1.f;
        sample_grad[c] += g * scale;
    }
}

void SoftmaxCrossEntropy::backward() {
    if (!input_->requires_grad())
        return;

    const auto& logits = input_->data();
    const auto& out_grad = grad();
    auto& in_grad = input_->grad();

    softmax_cross_entropy_backward_kernel<<<logits.cols(), get_block_size()>>>(
        probs_.data(),
        labels_->data().data(),
        out_grad.data(),
        in_grad.data(),
        logits.cols(),
        logits.rows()
    );

    CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::nn::layer
