#ifndef RELU_LAYER_H
#define RELU_LAYER_H
#include "layer.h"
class ReLULayer : public Layer  {
public:
  ReLULayer(float negative_slope = 0.0f);
  // input:
  //   previous_destination_dimensions
  // outputs:
  //   destination_dimensions
  //   fwd_p, forward pass primitive
  //   bwd_p, forward pass primitive
  //   requested_resources
  void createPrimitives(std::vector<size_t> const &src_dimensions,
                        std::vector<size_t> &dst_dimensions,
                        std::vector<dnnPrimitive_t> &fwd_p,
                        std::vector<dnnPrimitive_t> &bwd_p);
  size_t getNumberOfFwdPrimitives();
  size_t getNumberOfBwdPrimitives();
  bool needsPadding(std::vector<size_t> &padding_size);
  std::string getDebugString() const;
private:
  float negative_slope_;
};
#endif // RELU_LAYER_H
