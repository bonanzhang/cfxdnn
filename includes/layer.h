#ifndef LAYER_H
#define LAYER_H
#include "mkl.h"
#include <vector>
class Layer {
public:
  // input: previous_destination_dimensions
  // outputs:
  // destination_dimensions
  // p, forward pass primitive
  virtual void createForwardPrimitive(std::vector<size_t> previous_destination_dimensions,
                                      std::vector<size_t> destination_dimensions,
                                      dnnPrimitive_t *p);
  // input: next_source_dimensions
  // outputs:
  // source_dimensions
  // p, backward pass primitive
  virtual void createBackwardPrimitive(std::vector<size_t> next_source_dimensions,
                                       std::vector<size_t> source_dimensions,
                                       dnnPrimitive_t *p);
};
#endif // LAYER_H
