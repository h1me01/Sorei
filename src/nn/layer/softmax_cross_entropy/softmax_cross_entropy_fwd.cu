#include "softmax_cross_entropy.h"

namespace sorei::nn::layer {

// inspired from https://github.com/Infatoshi/mnist-cuda
__global__ void softmax_cross_entropy_forward_kernel(
    const float* logits,
    const int* labels,
    float* losses,
    float* probs,
    int batch_size,
    int num_classes
) {
    extern __shared__ float shared_logits[];

    const int sample = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    if (sample >= batch_size)
        return;

    const float* sample_logits = logits + sample * num_classes;
    float* sample_probs = probs + sample * num_classes;

    if (tid < num_classes)
        shared_logits[tid] = sample_logits[tid];
    __syncthreads();

    __shared__ float max_logit;
    __shared__ float sum_exp;

    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < num_classes)
            shared_logits[tid] = max(shared_logits[tid], shared_logits[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        max_logit = shared_logits[0];
    __syncthreads();

    if (tid < num_classes)
        shared_logits[tid] = expf(sample_logits[tid] - max_logit);
    __syncthreads();

    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < num_classes)
            shared_logits[tid] += shared_logits[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        sum_exp = shared_logits[0];
    __syncthreads();

    if (tid < num_classes) {
        float prob = expf(sample_logits[tid] - max_logit) / sum_exp;

        sample_probs[tid] = prob;

        if (tid == labels[sample])
            losses[sample] = -logf(max(prob, 1e-7f));
    }
}

void SoftmaxCrossEntropy::forward() {
    const auto& logits = input_->data();
    const auto& labels = labels_->data();
    auto& losses = data();

    losses.resize(shape());
    probs_.resize(logits.shape());

    const int threads = get_block_size();
    const size_t smem = threads * sizeof(float);

    softmax_cross_entropy_forward_kernel<<<logits.cols(), threads, smem>>>(
        logits.data(), labels.data(), losses.data(), probs_.data(), logits.cols(), logits.rows()
    );

    CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace sorei::nn::layer
