#ifndef SEQUENTIAL_NETWORK_H
#define SEQUENTIAL_NETWORK_H
#include "primitive.h"
// A container wrapper around the MKL DNN primitives
// Each of the primitives is wrapped in a "Layer"
// This is a collection of those layers
// and a set of operations on every one of those.
// non-obvious logic:
// each layer has a primitive, which acts on some resources
// these resources are divided among the layers and the network
// for example, the network holds the data buffers between layers
// and the layers themselves hold their own weights and gradients
class SequentialNetwork {
  public:
    SequentialNetwork(size_t batch_size, size_t channel, size_t height, size_t width);
    ~SequentialNetwork();
    int add_layer(Layer *l);
    void finalize_layers();
    void train();
    void forward();
    void backward();
    void update(Optimizer *opt);
  private:
    std::vector<Layer *> layers_;
    std::vector<Primitive *> net_;
    size_t batch_size_;
    size_t channel_;
    size_t height_;
    size_t width_;
    std::vector<void *> data_tensors_;
    std::vector<void *> gradient_tensors_;
};
#endif // SEQUENTIAL_NETWORK_H
