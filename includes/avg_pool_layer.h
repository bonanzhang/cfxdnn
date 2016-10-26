#ifndef AVG_POOL_LAYER_H
#define AVG_POOL_LAYER_H
#include "layer.h"
#include <cmath>
class AvgPoolLayer : public Layer  {
public:
  AvgPoolLayer(size_t kernel_w,
               size_t kernel_h,
               size_t stride_w,
               size_t stride_h,
               size_t padding_w,
               size_t padding_h
               );
  // input:
  //   previous_destination_dimensions
  // outputs:
  //   destination_dimensions
  //   fwd_p, forward pass primitive
  //   bwd_p, forward pass primitive
  void createPrimitives(std::vector<size_t> const &src_dimensions,
                        std::vector<size_t> &dst_dimensions,
                        std::vector<dnnPrimitive_t> &fwd_p,
                        std::vector<dnnPrimitive_t> &bwd_p);
  size_t getNumberOfFwdPrimitives() const;
  size_t getNumberOfBwdPrimitives() const;
  std::string getDebugString() const;
private:
  size_t kernel_w_;
  size_t kernel_h_;
  size_t stride_w_;
  size_t stride_h_;
  size_t padding_w_;
  size_t padding_h_;
};
#endif // AVG_POOL_LAYER_H
