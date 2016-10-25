#include "convolution_layer.h"
#include <iostream>
//TODO warnings based on the input
//  - padding > kernel_size/2 (convolution outside...)
//  - check that output_c_ is > 0
ConvolutionLayer::ConvolutionLayer(size_t kernel_w, size_t kernel_h,
                                   size_t stride_w, size_t stride_h,
                                   size_t padding_w, size_t padding_h,
                                   size_t output_c, bool bias) 
  : kernel_w_(kernel_w), kernel_h_(kernel_h),
    stride_w_(stride_w), stride_h_(stride_h),
    padding_w_(padding_w), padding_h_(padding_h),
    output_c_(output_c), bias_(bias) { }
void ConvolutionLayer::createPrimitives(std::vector<size_t> const &src_dimensions,
                                 std::vector<size_t> &dst_dimensions,
                                 std::vector<dnnPrimitive_t> &fwd_p,
                                 std::vector<dnnPrimitive_t> &bwd_p) {
  size_t const dimension = src_dimensions.size();
  for (auto i : src_dimensions) {
    std::cout << i << " ";
  } std::cout << std::endl;
  // TODO: Check dimensions
  // Computing Dimensions. Convolution does not change size. 
  size_t dst_w = std::ceil(((float) (src_dimensions[0]-kernel_w_+2*padding_w_))/stride_w_)+1;
  size_t dst_h = std::ceil(((float) (src_dimensions[1]-kernel_h_+2*padding_h_))/stride_h_)+1;
  dst_dimensions.push_back(dst_w); 
  dst_dimensions.push_back(dst_h); 
  dst_dimensions.push_back(output_c_); 
  dst_dimensions.push_back(src_dimensions[3]); 
  for (auto i : dst_dimensions) {
    std::cout << i << " ";
  } std::cout << std::endl;
  // Making a copy of input and output dims because the primitive
  // needs size_t*. Also computing the strides for layout
  size_t src_dim_arr[dimension], dst_dim_arr[dimension];
  for(int i = 0; i < dimension; i++) { 
    src_dim_arr[i] = src_dimensions[i]; 
    dst_dim_arr[i] = dst_dimensions[i]; 
  }
  size_t kernel_size_arr[2]   = {kernel_w_, kernel_h_};
  size_t kernel_stride_arr[2] = {stride_w_, stride_h_};
  int input_offset_arr[2]  = {-static_cast<int>(padding_h_),
                              -static_cast<int>(padding_w_)};
  // Creating Convolution primitive. Link to MKL page on convolution primitive:
  // https://software.intel.com/en-us/node/684776
  // Creating the Primitives. Only one needed for each for Convolution
  if(bias_) {
    dnnConvolutionCreateForwardBias_F32(&fwd_p[0], NULL,
                                        dnnAlgorithmConvolutionDirect, dimension,
                                        src_dim_arr, dst_dim_arr,
                                        kernel_size_arr, kernel_stride_arr,
                                        input_offset_arr, dnnBorderZeros);
  } else {
    dnnConvolutionCreateForward_F32(&fwd_p[0], NULL,
                                    dnnAlgorithmConvolutionDirect, dimension,
                                    src_dim_arr, dst_dim_arr,
                                    kernel_size_arr, kernel_stride_arr,
                                    input_offset_arr, dnnBorderZeros);
  }
  dnnLayout_t dbg_layout;
  dnnLayoutCreateFromPrimitive_F32(&dbg_layout, fwd_p[0], dnnResourceSrc);
  std::cout << "conv src: " << dnnLayoutGetMemorySize_F32(dbg_layout) << std::endl;
  dnnLayoutCreateFromPrimitive_F32(&dbg_layout, fwd_p[0], dnnResourceDst);
  std::cout << "conv dst: " << dnnLayoutGetMemorySize_F32(dbg_layout) << std::endl;
  // Primitive input output weight buffer size debug messages
  dnnConvolutionCreateBackwardData_F32(&bwd_p[0], NULL,
                                       dnnAlgorithmConvolutionDirect, dimension,
                                       src_dim_arr, dst_dim_arr,
                                       kernel_size_arr, kernel_stride_arr,
                                       input_offset_arr, dnnBorderZeros);
  dnnLayoutCreateFromPrimitive_F32(&dbg_layout, bwd_p[0], dnnResourceDiffSrc);
  std::cout << "conv diff src: " << dnnLayoutGetMemorySize_F32(dbg_layout) << std::endl;
  dnnLayoutCreateFromPrimitive_F32(&dbg_layout, bwd_p[0], dnnResourceDiffDst);
  std::cout << "conv diff dst: " << dnnLayoutGetMemorySize_F32(dbg_layout) << std::endl;
  dnnConvolutionCreateBackwardFilter_F32(&bwd_p[1], NULL,
                                         dnnAlgorithmConvolutionDirect, dimension,
                                         src_dim_arr, dst_dim_arr,
                                         kernel_size_arr, kernel_stride_arr,
                                         input_offset_arr, dnnBorderZeros);
  dnnLayoutCreateFromPrimitive_F32(&dbg_layout, bwd_p[1], dnnResourceDiffSrc);
  std::cout << "conv back filter diff src: " << dnnLayoutGetMemorySize_F32(dbg_layout) << std::endl;
  dnnLayoutCreateFromPrimitive_F32(&dbg_layout, bwd_p[1], dnnResourceDiffDst);
  std::cout << "conv back filter diff dst: " << dnnLayoutGetMemorySize_F32(dbg_layout) << std::endl;
  if (bias_) {
    dnnConvolutionCreateBackwardBias_F32(&bwd_p[2], NULL,
                                         dnnAlgorithmConvolutionDirect,
                                         dimension, dst_dim_arr);
  }
}
size_t ConvolutionLayer::getNumberOfFwdPrimitives() const {
  // Convolution has one forward primitive
  return 1;
}
size_t ConvolutionLayer::getNumberOfBwdPrimitives() const {
  // Convolution has two or three backward primitive
  return (bias_) ? 3 : 2;
} 
bool ConvolutionLayer::needsPadding(std::vector<size_t> &padding_size) const {
  padding_size.push_back(padding_w_);
  padding_size.push_back(padding_h_);
  return true;
}
std::string ConvolutionLayer::getDebugString() const {
  return std::string("ConvolutionLayer");
}
