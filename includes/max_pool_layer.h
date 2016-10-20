#ifndef MAX_POOL_LAYER_H
#define MAX_POOL_LAYER_H
#include "layer.h"
#include <cmath>
class MaxPoolLayer : public Layer  {
public:
  MaxPoolLayer(size_t kernel_w,
               size_t kernel_h,
               size_t stride_w,
               size_t stride_h,
               size_t padding_w,
               size_t padding_h
               );
  // input:
  //   src_dimensions
  // outputs:
  //   dst_dimensions
  //   fwd_p, forward pass primitive
  //   bwd_p, forward pass primitive
  //   requested_fwd_resources
  //   requested_bwd_resources
  void createPrimitives(std::vector<size_t> const &src_dimensions,
                        std::vector<size_t> &dst_dimensions,
                        std::vector<dnnPrimitive_t> &fwd_p,
                        std::vector<dnnPrimitive_t> &bwd_p);
  size_t getNumberOfFwdPrimitives();
  size_t getNumberOfBwdPrimitives();
  bool needsPadding(std::vector<size_t> &padding_size);
  std::string getDebugString() const;
private:
  size_t kernel_w_;
  size_t kernel_h_;
  size_t stride_w_;
  size_t stride_h_;
  size_t padding_w_;
  size_t padding_h_;
};
#endif // MAX_POOL_LAYER_H
