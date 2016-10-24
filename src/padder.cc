#include "padder.h"
#include <iostream>
Padder::Padder(std::vector<size_t> const &src_dimensions, 
               std::vector<size_t> const &padding_size, 
               std::vector<size_t> &dst_dimensions,
               bool unpad_backwards=false) 
    : src_dimensions_(src_dimensions),
      padding_size_(padding_size),
      unpad_backwards_(unpad_backwards) {
  // TODO: check that dimension is 4 and padding is 2
  std::copy(src_dimensions.begin(), src_dimensions.end(), std::back_inserter(dst_dimensions));
  for(int i = 0; i < padding_size.size(); i++) {
    dst_dimensions[i] += 2 * padding_size[i];
  }
}
// Forward Propagation for this layer.
void Padder::forward() {
//  std::cout << "execute forward with input: " << src_ << std::endl;
//  std::cout << "execute forward with output: " << dst_ << std::endl;
  const int ldd = src_dimensions_[0]+2*padding_size_[0];
  const int col_pad = padding_size_[1];
  const int row_pad = padding_size_[0];
  for(int i = 0; i < src_dimensions_[1]; i++) {
    for(int j = 0; j < src_dimensions_[0]; j++) {
      dst_[(i+col_pad)*ldd+(j+row_pad)] = src_[i*src_dimensions_[0]]; 
    }
  } 
}
// Backward Propagation for this layer.
void Padder::backward() {
  if(unpad_backwards_) {
//    std::cout << "execute backward with input: " << dst_ << std::endl;
//    std::cout << "execute backward with output: " << src_ << std::endl;
    int const ldd = src_dimensions_[0]+2*padding_size_[0];
    int const col_pad = padding_size_[1];
    int const row_pad = padding_size_[0];
    for(int i = 0; i < src_dimensions_[1]; i++) {
      for(int j = 0; j < src_dimensions_[0]; j++) {
        src_[i*src_dimensions_[0]] = dst_[(i+col_pad)*ldd+(j+row_pad)]; 
      }
    } 
  }
//TODO implements backwards
} 
// Updates weights of the layer based on the gradients.
void Padder::update(Optimizer const &opt, float learning_rate) {

}
// Fills the primitive's weights, if applicable
void Padder::initialize(Initializer const &ini) {

}
// "Connect" the layers in a neural network. This is done 
// automatically by network objects (e.g. sequencial_network)
void Padder::setFwdInput(void* src) {
  src_ = static_cast<float *>(src);
}
void Padder::setFwdOutput(void* dst) {
  dst_ = static_cast<float *>(dst);
  // Initialize it to 0 so this does not happen at every call to forward.
}
void Padder::setBwdInput(void* diffdst) {
  diffdst_ = static_cast<float *>(diffdst);
}
void Padder::setBwdOutput(void* diffsrc) {
  diffsrc_ = static_cast<float *>(diffsrc);
}
