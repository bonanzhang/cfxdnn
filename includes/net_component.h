#ifndef NET_COMPONENT_H
#define NET_COMPONENT_H
#include "optimizer.h"
#include "initializer.h"
// This is an interface
// it has the functions a layer in a DNN might call:
// this includes the obvious forward, backward, and update
// this also has the book keeping functions like
// keeping track of the input output buffers,
// and initializing weights and biases
//
// Everything here is pure virtual
// classes that implement this interface:
// Primitive (primitive.h)
// Padder    (padder.h)
class NetComponent {
  public:
    // Forward Propagation for this layer.
    virtual void forward() = 0;
    // Backward Propagation for this layer.
    virtual void backward() = 0; 
    // Updates weights of the layer based on the gradients.
    virtual void update(Optimizer const &opt, float learning_rate) = 0;
    // Fills the primitive's weights, if applicable
    virtual void initialize(Initializer const &ini) = 0;
    // "Connect" the layers in a neural network. This is done 
    // automatically by network objects (e.g. sequencial_network)
    virtual void setFwdInput(void* src) = 0;
    virtual void setFwdOutput(void* dst) = 0;
    virtual void setBwdInput(void* diffdst) = 0;
    virtual void setBwdOutput(void* diffsrc) = 0;
    // Get pointer to buffer (resource). Used to get, for 
    // example, the weights of a given layer
};
#endif // NET_COMPONENT_H
