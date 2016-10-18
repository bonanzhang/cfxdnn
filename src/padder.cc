#include "padder.h"
Padder::Padder(std::vector<size_t> const &src_dimensions, std::vector<size_t> const &padding_size, std::vector<size_t> &dst_dimensions , bool unpad_backwards=false) {
  
  src_dimensions_ = src_dimensions;
  padding_size_ = padding_size;
  unpad_backwards_ = unpad_backwards;

  std::copy(src_dimensions.begin(), src_dimensions.end(), std::back_inserter(dst_dimensions));
  for(int i = 0; i < padding_size.size(); i++) {
    dst_dimensions[i] += 2 * padding_size[i];
  }
}
    // Forward Propagation for this layer.
void Padder::forward() {

}
// Backward Propagation for this layer.
void Padder::backward() {

} 
// Updates weights of the layer based on the gradients.
void Padder::update(Optimizer* opt, float learning_rate) {

}
// Fills the primitive's weights, if applicable
void Padder::initialize(Initializer *ini) {

}
// "Connect" the layers in a neural network. This is done 
// automatically by network objects (e.g. sequencial_network)
void Padder::setFwdInput(void* src) {
  src_ = (float *) src;
}
void Padder::setFwdOutput(void* dst) {
  dst_ = (float *) dst;
}
void Padder::setBwdInput(void* diffdst) {
  diffdst_ = (float *) diffdst;
}
void Padder::setBwdOutput(void* diffsrc) {
  diffsrc_ = (float *) diffsrc;
}
