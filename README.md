# Sorei

A CUDA-accelerated neural network training library for C++20. Sorei provides a fluent graph-builder API for defining computation graphs with automatic differentiation.

## Requirements

- CMake 3.18+
- C++20-capable compiler
- CUDA Toolkit (cuBLAS)

## Building

Sorei is a static library used via CMake's `add_subdirectory`. To integrate it into a project:

```cmake
add_subdirectory(path/to/sorei sorei)
target_link_libraries(my_target PRIVATE sorei)
```

To build standalone:

```bash
cmake -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j
```

## Usage

Define a model by subclassing `sorei::nn::Model` and implementing `build_graph`:

```cpp
#include "sorei/nn.h"

struct MyModel : public sorei::nn::Model {
    sorei::nn::GraphOutput build_graph(sorei::nn::graph::GraphBuilder& b) override {
        auto x      = b.input_float({INPUT_DIM, 0}, "x");
        auto labels = b.input_int({1, 0}, "labels");

        auto l1 = b.affine_layer(INPUT_DIM, HIDDEN_DIM);
        auto l2 = b.affine_layer(HIDDEN_DIM, NUM_CLASSES);

        auto out  = l2(l1(x).relu());
        auto loss = out.softmax_cross_entropy(labels).mean();

        return {.prediction = out, .loss = loss};
    }
};
```

Then train with an optimizer and a learning-rate scheduler:

```cpp
MyModel model;
auto optim    = sorei::nn::optim::AdamW(model.params(), 0.9f, 0.999f, 0.01f);
auto lr_sched = sorei::nn::lr_sched::CosineAnnealing(lr, lr * 0.1f, epochs);

for (int epoch = 0; epoch < epochs; ++epoch) {
    model.forward({{"x", inputs}, {"labels", targets}});
    model.backward();
    optim.step(lr_sched.get());
    lr_sched.step();
}
```

## Examples

| Example | Description |
|---------|-------------|
| [`examples/mnist`](examples/mnist) | MLP trained on MNIST handwritten digits |
| [`examples/astra`](examples/astra) | NNUE for my chess engine [Astra](https://github.com/h1me01/Astra) |

### Running an Example

Build and run one of the examples:

```bash
cd examples/mnist
cmake -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j
./build/release/mnist_example
```

## License

MIT — see [LICENSE](LICENSE).
