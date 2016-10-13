#include "relu_layer.h"
ReLULayer::ReLULayer(float negative_slope) {
  negative_slope_ = negative_slope;
}
void ReLULayer::createPrimitives(std::vector<size_t> const &src_dimensions,
                                 std::vector<size_t> &dst_dimensions,
                                 dnnPrimitive_t *fwd_p,
                                 dnnPrimitive_t *bwd_p,
                                 std::vector<dnnResourceType_t> &requested_fwd_resources,
                                 std::vector<dnnResourceType_t> &requested_bwd_resources) {
  const size_t dimension = src_dimensions.size();
  /* TODO: if(dimension != 4) { do something } */
  // Computing Dimensions. ReLU does not change size. 
  std::copy(src_dimensions.begin(), src_dimensions.end(), dst_dimensions.begin());
  
  // Making a copy of input and output dims because the primitive
  // needs size_t*. Also computing the strides for layout
  size_t src_dimensions_[dimension], dst_dimensions_[dimension];
  for(int i = 0; i < dimension; i++) { 
    src_dimensions_[i] = src_dimensions[i]; 
    dst_dimensions_[i] = dst_dimensions[i]; 
  }
  size_t src_strides_[4] = {1,
                            src_dimensions_[0],
                            src_dimensions_[0]*src_dimensions_[1],
                            src_dimensions_[0]*src_dimensions_[1]*src_dimensions_[2]};
  size_t dst_strides_[4] = {1,
                            dst_dimensions_[0],
                            dst_dimensions_[0]*dst_dimensions_[1],
                            dst_dimensions_[0]*dst_dimensions_[1]*dst_dimensions_[2]};
  // Creating Layouts needed for primitives
  dnnLayout_t src_layout;
  dnnLayoutCreate_F32(&src_layout, dimension, src_dimensions_, src_strides_);
  dnnLayout_t dst_layout;
  dnnLayoutCreate_F32(&dst_layout, dimension, dst_dimensions_, dst_strides_);
  // Creating the Primitives
  dnnReLUCreateForward_F32(fwd_p, NULL, src_layout, negative_slope_);
  dnnReLUCreateBackward_F32(bwd_p, NULL, dst_layout, src_layout, negative_slope_);
  // Deleting the Layouts
  dnnLayoutDelete_F32(dst_layout);
  dnnLayoutDelete_F32(src_layout);

  // No Requested Resource
}
