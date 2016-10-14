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
//
// example usage:
// SequentialNetwork net;
// net.add(new ConvLayer(...));
// net.add(new ReLULayer(...));
// net.add(new MaxPLayer(...));
class SequentialNetwork {
  public:
    SequentialNetwork(size_t batch_size, size_t channel, size_t height, size_t width);
    ~SequentialNetwork();
    // each time you add a layer, you get the 0-indexed id of that layer
    int add_layer(Layer *l);
    // must call this after adding all the layers
    // finalize also will initialize all the weights
    void finalize_layers();
    //training with 
    void train(void *X, void *y, Optimizer *o);
    void forward();
    void backward();
    void update(Optimizer *opt, float learning_rate);
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
