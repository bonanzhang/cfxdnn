#include "convolution_layer.h"
ConvolutionLayer::ConvolutionLayer(size_t kernel_w,
                            size_t kernel_h,
                            size_t stride_w,
                            size_t stride_h,
                            size_t padding_w,
                            size_t padding_h,
                            size_t output_c,
                            bool bias) {
  kernel_w_  = kernel_w;
  kernel_h_  = kernel_h;
  stride_w_  = stride_w;
  stride_h_  = stride_h;
  padding_w_ = padding_w;
  padding_h_ = padding_h;
  output_c_  = output_c;
  bias_      = bias;

  //TODO warnings based on the niput
  //  - padding > kernel_size/2 (convolution outside...)
  //  - check that output_c_ is > 0
}
void ConvolutionLayer::createPrimitives(std::vector<size_t> const &src_dimensions,
                                 std::vector<size_t> &dst_dimensions,
                                 std::vector<dnnPrimitive_t> &fwd_p,
                                 std::vector<dnnPrimitive_t> &bwd_p,
                                 std::vector<std::vector<dnnResourceType_t>> &requested_fwd_resources,
                                 std::vector<std::vector<dnnResourceType_t>> &requested_bwd_resources) {

  const size_t dimension = src_dimensions.size();
  // TODO: Check dimensions
  // Computing Dimensions. Convolution does not change size. 

  size_t dst_w = std::ceil(((float) (src_dimensions[0]-kernel_w_+2*padding_w_))/stride_w_)+1;
  size_t dst_h = std::ceil(((float) (src_dimensions[1]-kernel_h_+2*padding_h_))/stride_h_)+1;
  dst_dimensions.push_back(dst_w); 
  dst_dimensions.push_back(dst_h); 
  dst_dimensions.push_back(output_c_); 
  dst_dimensions.push_back(src_dimensions[3]); 
  // Making a copy of input and output dims because the primitive
  // needs size_t*. Also computing the strides for layout
  size_t src_dimensions_[dimension], dst_dimensions_[dimension];
  for(int i = 0; i < dimension; i++) { 
    src_dimensions_[i] = src_dimensions[i]; 
    dst_dimensions_[i] = dst_dimensions[i]; 
  }

  size_t kernel_size_[2]   = {kernel_w_, kernel_h_};
  size_t kernel_stride_[2] = {stride_w_, stride_h_};
  int input_offset_[2]  = {-padding_w_, -padding_h_};
 

  // Creating Convolution primitive. Link to MKL page on convolution primitive:
  // https://software.intel.com/en-us/node/684776
  // Creating the Primitives. Only one needed for each for Convolution
  if(bias_) {
    dnnConvolutionCreateForwardBias_F32(&fwd_p[0], NULL, dnnAlgorithmConvolutionDirect, dimension, src_dimensions_, dst_dimensions_, kernel_size_, kernel_stride_, input_offset_, dnnBorderZeros);
    // Requested Fwd Resource for Convolution
    requested_fwd_resources[0].push_back(dnnResourceFilter);
    requested_fwd_resources[0].push_back(dnnResourceBias);

    dnnConvolutionCreateBackwardData_F32(&bwd_p[0], NULL, dnnAlgorithmConvolutionDirect, dimension, src_dimensions_, dst_dimensions_, kernel_size_, kernel_stride_, input_offset_, dnnBorderZeros);
    dnnConvolutionCreateBackwardFilter_F32(&bwd_p[1], NULL, dnnAlgorithmConvolutionDirect, dimension, src_dimensions_, dst_dimensions_, kernel_size_, kernel_stride_, input_offset_, dnnBorderZeros);
    dnnConvolutionCreateBackwardBias_F32(&bwd_p[2], NULL, dnnAlgorithmConvolutionDirect, dimension, dst_dimensions_);
    // Requested Bwd Resource for Convolution
    requested_bwd_resources[1].push_back(dnnResourceDiffFilter);
    requested_bwd_resources[2].push_back(dnnResourceDiffBias);

  } else {
    int e = dnnConvolutionCreateForward_F32(&fwd_p[0], NULL, dnnAlgorithmConvolutionDirect, dimension, src_dimensions_, dst_dimensions_, kernel_size_, kernel_stride_, input_offset_, dnnBorderZeros);
    // Requested Fwd Resource for Convolution
    requested_fwd_resources[0].push_back(dnnResourceFilter);

    dnnConvolutionCreateBackwardData_F32(&bwd_p[0], NULL, dnnAlgorithmConvolutionDirect, dimension, src_dimensions_, dst_dimensions_, kernel_size_, kernel_stride_, input_offset_, dnnBorderZeros);
    dnnConvolutionCreateBackwardFilter_F32(&bwd_p[1], NULL, dnnAlgorithmConvolutionDirect, dimension, src_dimensions_, dst_dimensions_, kernel_size_, kernel_stride_, input_offset_, dnnBorderZeros);
    // Requested Bwd Resource for Convolution
    requested_bwd_resources[1].push_back(dnnResourceDiffFilter);
  }
}

size_t ConvolutionLayer::getNumberOfFwdPrimitives() {
  // Convolution has one forward primitive
  return 1;
}
size_t ConvolutionLayer::getNumberOfBwdPrimitives() {
  // Convolution has two or three backward primitive
  return (bias_) ? 3 : 2;
} 
bool ConvlutionLayer::needsPadding() {
  return true;
}

