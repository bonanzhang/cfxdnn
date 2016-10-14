#ifndef PRIMITIVE_H
#define PRIMITIVE_H
#include <mkl.h>
#include <vector>
#include "optimizer.h"
#include "layer.h"
// TODO: fix this paragraph
// Parent Class for all neural network layers. 
// This class contains the information about the parameters
// required toreconstruct the layer (e.g. kernel size for 
// convolution), and also contains the data (such as filter 
// weights) of the layers.

// Generally the user should only have to create the layer
// when defining a network, all tasks like initialization 
// or forward tasks will be done by the network object

// All functions here are safe to call by derived classes.
// If it is not applicable (e.g. update() on ReLU) it will
// simply do nothing.
class Primitive {
  public:
    Primitive(Layer *l,
              std::vector<size_t> const &input_dimensions,
              std::vector<size_t> &output_dimensions);
    ~Primitive();
    // Forward Propagation for this layer.
    void forward();
    // Backward Propagation for this layer.
    void backward(); 
    // Updates weights of the layer based on the gradients.
    void update(Optimizer* opt, float learning_rate);
    // "Connect" the layers in a neural network. This is done 
    // automatically by network objects (e.g. sequencial_network)
    void setFwdInput(void* src);
    void setFwdOutput(void* dst);
    void setBwdInput(void* diffdst);
    void setBwdOutput(void* diffsrc);
    // Get pointer to buffer (resource). Used to get, for 
    // example, the weights of a given layer
    void* getResource(dnnResourceType_t type);
  protected:
    // dnnPrimitives is the Intel MKL computational kernel 
    std::vector<dnnPrimitive_t> forward_p;
    std::vector<dnnPrimitive_t> backward_p;
    // Contains the resources. Use getResource() to access.
    void* resources[dnnResourceNumber];
    size_t resource_sizes[dnnResourceNumber];
    std::vector<std::vector<dnnResourceType_t>> requested_fwd_resources;
    std::vector<std::vector<dnnResourceType_t>> requested_bwd_resources;
};
#endif // PRIMITIVE_H
