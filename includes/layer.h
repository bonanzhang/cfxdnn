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
                                std::vector<dnnPrimitive_t> &fwd_p,
                                std::vector<dnnPrimitive_t> &bwd_p,
                                std::vector<std::vector<dnnResourceType_t>> &requested_fwd_resources,
                                std::vector<std::vector<dnnResourceType_t>> &requested_bwd_resources
                                ); 
  // Returns the number of primitives needed
  virtual size_t getNumberOfFwdPrimitives();
  virtual size_t getNumberOfBwdPrimitives();
};
#endif // LAYER_H
