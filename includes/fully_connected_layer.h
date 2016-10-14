#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H
#include "layer.h"
class FullyConnectedLayer : public Layer  {
public:
  // input:
  //   output_channels, number of output channels
  FullyConnectedLayer(size_t output_channels, bool bias);

  // input:
  //   previous_destination_dimensions
  // outputs:
  //   destination_dimensions
  //   fwd_p, forward pass primitives
  //   bwd_p, forward pass primitives
  //   requested_fwd_resources
  //   requested_bwd_resources
  void createPrimitives(std::vector<size_t> const &src_dimensions, 
                        std::vector<size_t> &dst_dimensions, 
                        std::vector<dnnPrimitive_t> &fwd_p,
                        std::vector<dnnPrimitive_t> &bwd_p,
                        std::vector<std::vector<dnnResourceType_t>> &requested_fwd_resources,
                        std::vector<std::vector<dnnResourceType_t>> &requested_bwd_resources
                        ); 
  size_t getNumberOfFwdPrimitives();
  size_t getNumberOfBwdPrimitives();
private:
  size_t output_channels_;
  bool bias_;
};
#endif // FULLY_CONNECTED_LAYER_H
