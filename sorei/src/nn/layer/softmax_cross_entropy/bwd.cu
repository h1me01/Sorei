#include "softmax_cross_entropy.h"

namespace sorei::nn {

template <bool Overwrite>
__global__ void softmax_cross_entropy_bwd_kernel(
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
        if constexpr (Overwrite)
            sample_grad[c] = g * scale;
        else
            sample_grad[c] += g * scale;
    }
}

void SoftmaxCrossEntropy::backward() {
    auto& in_g = input_->grad();
    if (in_g.empty())
        return;

    const auto& logits = input_->data();
    const auto& out_g = grad();

    if (input_->consume_grad_write()) {
        softmax_cross_entropy_bwd_kernel<true><<<logits.cols(), get_block_size()>>>(
            probs_.data(),
            labels_->data().data(),
            out_g.data(),
            in_g.data(),
            logits.cols(),
            logits.rows()
        );
    } else {
        softmax_cross_entropy_bwd_kernel<false><<<logits.cols(), get_block_size()>>>(
            probs_.data(),
            labels_->data().data(),
            out_g.data(),
            in_g.data(),
            logits.cols(),
            logits.rows()
        );
    }

    SOREI_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::nn
