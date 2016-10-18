#include "max_pool_layer.h"
#include <iostream>
MaxPoolLayer::MaxPoolLayer(size_t kernel_w,
                            size_t kernel_h,
                            size_t stride_w,
                            size_t stride_h,
                            size_t padding_w,
                            size_t padding_h) {
  kernel_w_  = kernel_w;
  kernel_h_  = kernel_h;
  stride_w_  = stride_w;
  stride_h_  = stride_h;
  padding_w_ = padding_w;
  padding_h_ = padding_h;

  //TODO warnings based on the niput
  //  - padding > kernel_size/2 (max pool outside...)
}
void MaxPoolLayer::createPrimitives(std::vector<size_t> const &src_dimensions,
                                 std::vector<size_t> &dst_dimensions,
                                 std::vector<dnnPrimitive_t> &fwd_p,
                                 std::vector<dnnPrimitive_t> &bwd_p,
                                 std::vector<std::vector<dnnResourceType_t>> &requested_fwd_resources,
                                 std::vector<std::vector<dnnResourceType_t>> &requested_bwd_resources) {
  dnnError_t e;
  size_t const dimension = src_dimensions.size();
  // TODO: Check dimensions
  // Computing Dimensions. MaxPool does not change size. 

  size_t dst_w = std::ceil(((float) (src_dimensions[0]-kernel_w_+2*padding_w_))/stride_w_)+1;
  size_t dst_h = std::ceil(((float) (src_dimensions[1]-kernel_h_+2*padding_h_))/stride_h_)+1;
  dst_dimensions.push_back(dst_w); 
  dst_dimensions.push_back(dst_h); 
  dst_dimensions.push_back(src_dimensions[2]); 
  dst_dimensions.push_back(src_dimensions[3]); 
 
  // Making a copy of input and output dims because the primitive
  // needs size_t*. Also computing the strides for layout
  size_t src_dimensions_[dimension], dst_dimensions_[dimension];
  for(int i = 0; i < dimension; i++) { 
    src_dimensions_[i] = src_dimensions[i]; 
    dst_dimensions_[i] = dst_dimensions[i]; 
  }
  //Computing strides
  size_t src_strides_[dimension], dst_strides_[dimension];
  size_t src_stride=1,dst_stride=1;
  for(int i = 0; i < dimension; i++) { 
    src_strides_[i] = src_stride;
    src_stride *= src_dimensions[i]; 
    dst_strides_[i] = dst_stride;
    dst_stride *= dst_dimensions[i]; 
  }

  size_t kernel_size_[2]   = {kernel_h_, kernel_w_};
  size_t kernel_stride_[2] = {stride_h_, stride_w_};
  int input_offset_[2]  = {-padding_h_, -padding_w_};
 
  // Creating MaxPooling primitive. Link to MKL page on Pooling primitive:
  // https://software.intel.com/en-us/node/684776
  // Creating Layouts needed
  dnnLayout_t src_layout;
  e = dnnLayoutCreate_F32(&src_layout, dimension, src_dimensions_, src_strides_);
  if (e != E_SUCCESS) std::cout << "src layout failed\n";
  dnnLayout_t dst_layout;
  e = dnnLayoutCreate_F32(&dst_layout, dimension, dst_dimensions_, dst_strides_);
  if (e != E_SUCCESS) std::cout << "dst layout failed\n";

  // Creating the Primitives. Only one needed for each for MaxPool
  e = dnnPoolingCreateForward_F32(&fwd_p[0], NULL, dnnAlgorithmPoolingMax, src_layout, kernel_size_, kernel_stride_, input_offset_, dnnBorderZeros);
  if (e != E_SUCCESS) std::cout << "maxp forward failed\n";
  e = dnnPoolingCreateBackward_F32(&bwd_p[0], NULL, dnnAlgorithmPoolingMax, src_layout, kernel_size_, kernel_stride_, input_offset_, dnnBorderZeros);
  if (e != E_SUCCESS) std::cout << "maxp backward failed\n";
  
  // Deleting the Layouts
  dnnLayoutDelete_F32(dst_layout);
  dnnLayoutDelete_F32(src_layout);
  
  // Requested Resource for MaxPool
  requested_fwd_resources[0].push_back(dnnResourceWorkspace);
}

size_t MaxPoolLayer::getNumberOfFwdPrimitives() {
  // MaxPool has one forward primitive
  return 1;
}
size_t MaxPoolLayer::getNumberOfBwdPrimitives() {
  // MaxPool has one backward primitive
  return 1;
} 
bool MaxPoolLayer::needsPadding(std::vector<size_t> &padding_size) {
  return false;
}
