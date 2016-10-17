#ifndef NET_COMPONENTS_H
#define NET_COMPONENTS_H
#include "optimizer.h"
#include "initializer.h"
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
class NetComponents {
  public:
    // Forward Propagation for this layer.
    virtual void forward();
    // Backward Propagation for this layer.
    virtual void backward(); 
    // Updates weights of the layer based on the gradients.
    virtual void update(Optimizer* opt, float learning_rate);
    // Fills the primitive's weights, if applicable
    virtual void initialize(Initializer *ini);
    // "Connect" the layers in a neural network. This is done 
    // automatically by network objects (e.g. sequencial_network)
    virtual void setFwdInput(void* src);
    virtual void setFwdOutput(void* dst);
    virtual void setBwdInput(void* diffdst);
    virtual void setBwdOutput(void* diffsrc);
    // Get pointer to buffer (resource). Used to get, for 
    // example, the weights of a given layer
};
#endif // NET_COMPONENTS_H
