#ifndef PADDER_H
#define PADDER_H
#include "net_component.h"
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
class Padder : public NetComponent {
  public:
    Padder(std::vector<size_t> const &src_dims, std::vector<size_t> const &padding_size, std::vector<size_t> &dst_dimensions, bool unpad_backwards);
    // Forward Propagation for this layer.
    void forward();
    // Backward Propagation for this layer.
    void backward(); 
    // Updates weights of the layer based on the gradients.
    void update(Optimizer* opt, float learning_rate);
    // Fills the primitive's weights, if applicable
    void initialize(Initializer *ini);
    // "Connect" the layers in a neural network. This is done 
    // automatically by network objects (e.g. sequencial_network)
    void setFwdInput(void* src);
    void setFwdOutput(void* dst);
    void setBwdInput(void* diffdst);
    void setBwdOutput(void* diffsrc);
  private:
    std::vector<size_t> src_dimensions_;
    std::vector<size_t> padding_size_;
    float *src_; 
    float *dst_; 
    float *diffsrc_; 
    float *diffdst_; 
    bool unpad_backwards_;
};
#endif // PADDER_H
