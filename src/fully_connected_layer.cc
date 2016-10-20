#include "fully_connected_layer.h"
#include <iostream>
FullyConnectedLayer::FullyConnectedLayer(size_t output_channels, bool bias) {
  output_channels_ = output_channels;
  bias_ = bias;
}
void FullyConnectedLayer::createPrimitives(std::vector<size_t> const &src_dimensions,
                                 std::vector<size_t> &dst_dimensions,
                                 std::vector<dnnPrimitive_t> &fwd_p,
                                 std::vector<dnnPrimitive_t> &bwd_p,
                                 std::vector<std::vector<dnnResourceType_t>> &requested_fwd_resources,
                                 std::vector<std::vector<dnnResourceType_t>> &requested_bwd_resources) {
  size_t const dimension = src_dimensions.size();
  // Computing Dimensions. FullyConnected does not change size. 
  dst_dimensions.push_back(output_channels_);
  dst_dimensions.push_back(src_dimensions[dimension-1]); 
  // Making a copy of input and output dims because the primitive
  // needs size_t*. 
  size_t src_dimensions_[dimension]; 
  for(int i = 0; i < dimension; i++) { 
    src_dimensions_[i] = src_dimensions[i]; 
  }
  // Creating the Primitives
  if(bias_) {  // With bias
    // Pirimitives are carried out in order. 
    // For FC fwd: 0->forward
    dnnInnerProductCreateForwardBias_F32(&fwd_p[0], NULL, dimension, src_dimensions_, output_channels_);
    requested_fwd_resources[0].push_back(dnnResourceFilter);
    requested_fwd_resources[0].push_back(dnnResourceBias);

    // For FC bwd: 0->backward, 1->filter, 2-> bias
    dnnInnerProductCreateBackwardData_F32(&bwd_p[0], NULL, dimension, src_dimensions_, output_channels_);
    dnnInnerProductCreateBackwardFilter_F32(&bwd_p[1], NULL, dimension, src_dimensions_, output_channels_);
    requested_bwd_resources[1].push_back(dnnResourceDiffFilter);
    dnnInnerProductCreateBackwardBias_F32(&bwd_p[2], NULL, dimension, src_dimensions_);
    requested_bwd_resources[2].push_back(dnnResourceDiffBias);

  } else {
    // For FC fwd: 0->forward
    dnnInnerProductCreateForward_F32(&fwd_p[0], NULL, dimension, src_dimensions_, output_channels_);
    requested_fwd_resources[0].push_back(dnnResourceFilter);
    // Primitive input output buffer size debug messages
    dnnLayout_t dbg_layout;
    dnnLayoutCreateFromPrimitive_F32(&dbg_layout, fwd_p[0], dnnResourceSrc);
    std::cout << "fc src: " << dnnLayoutGetMemorySize_F32(dbg_layout) << std::endl;
    dnnLayoutCreateFromPrimitive_F32(&dbg_layout, fwd_p[0], dnnResourceDst);
    std::cout << "fc dst: " << dnnLayoutGetMemorySize_F32(dbg_layout) << std::endl;
    dnnLayoutCreateFromPrimitive_F32(&dbg_layout, bwd_p[0], dnnResourceDiffSrc);
    std::cout << "fc diff src: " << dnnLayoutGetMemorySize_F32(dbg_layout) << std::endl;
    dnnLayoutCreateFromPrimitive_F32(&dbg_layout, bwd_p[0], dnnResourceDiffDst);
    std::cout << "fc diff dst: " << dnnLayoutGetMemorySize_F32(dbg_layout) << std::endl;

    // For FC bwd: 0->backward, 1->filter
    dnnInnerProductCreateBackwardData_F32(&bwd_p[0], NULL, dimension, src_dimensions_, output_channels_);
    dnnInnerProductCreateBackwardFilter_F32(&bwd_p[1], NULL, dimension, src_dimensions_, output_channels_);
    requested_bwd_resources[1].push_back(dnnResourceDiffFilter);
  }
}


size_t FullyConnectedLayer::getNumberOfFwdPrimitives() {
  return 1;
}
size_t FullyConnectedLayer::getNumberOfBwdPrimitives() {
  return (bias_) ? 3 : 2;
}
bool FullyConnectedLayer::needsPadding(std::vector<size_t> &padding_size) {
  return false;
}
