#ifndef LAYER_H
#define LAYER_H
#include "mkl.h"
#include <vector>
class Layer {
public:
  // input:
  //   previous_destination_dimensions
  // outputs:
  //   destination_dimensions
  //   fwd_p, forward pass primitive
  //   bwd_p, forward pass primitive
  //   requested_resources
  virtual void createPrimitives(std::vector<size_t> const &src_dimensions, 
                                std::vector<size_t> &dst_dimensions, 
                                dnnPrimitive_t *fwd_p,
                                dnnPrimitive_t *bwd_p,
                                std::vector<dnnResourceType_t> &requested_resources);
};
#endif // LAYER_H
