#ifndef SEQUENTIAL_NETWORK_H
#define SEQUENTIAL_NETWORK_H
#include "layer.h"
#include <vector>
// A container wrapper around the MKL DNN primitives
// Each of the primitives is wrapped in a "Layer"
// This is a collection of those layers
// and a set of operations on every one of those.
class SequentialNetwork {
  private:
    std::vector<Layer *> net_;
  public:
    // add a layer to the network
    // the network is ordered like a queue:
    // the first layer added runs first
    void add_layer(Layer *l);
    // trains for some number of iterations
    void train();
    // the next three functions should be called one after another
    // in a loop
    // they are individually exposed for analysis of intermediate states
    // like the weights
    void forward();
    void backward();
    void update();
};
#endif // SEQUENTIAL_NETWORK_H
