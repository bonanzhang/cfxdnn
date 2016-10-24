#ifndef LAYER_H
#define LAYER_H
#include "mkl_dnn.h"
#include <vector>
#include <string>
class Layer {
public:
  virtual ~Layer() { }
  // input:
  //   previous_destination_dimensions
  // outputs:
  //   destination_dimensions
  //   fwd_p, forward pass primitive
  //   bwd_p, forward pass primitive
  virtual void createPrimitives(std::vector<size_t> const &src_dimensions, 
                                std::vector<size_t> &dst_dimensions, 
                                std::vector<dnnPrimitive_t> &fwd_p,
                                std::vector<dnnPrimitive_t> &bwd_p) = 0; 
  // Returns the number of primitives needed
  virtual size_t getNumberOfFwdPrimitives() const = 0;
  virtual size_t getNumberOfBwdPrimitives() const = 0;
  virtual bool needsPadding(std::vector<size_t> &padding_size) const = 0;
  virtual std::string getDebugString() const = 0;
};
#endif // LAYER_H
