#ifndef RELU_LAYER_H
#define RELU_LAYER_H
#include "layer.h"
class ReLULayer : public Layer  {
  public:
    ReLULayer(float negative_slope = 0.0f);
    // input: previous_destination_dimensions
    // outputs:
    // destination_dimensions
    // p, forward pass primitive
    void createForwardPrimitive(std::vector<size_t> previous_destination_dimensions,
                                        std::vector<size_t> destination_dimensions,
                                        dnnPrimitive_t *p);
    // input: next_source_dimensions
    // outputs:
    // source_dimensions
    // p, backward pass primitive
    void createBackwardPrimitive(std::vector<size_t> next_source_dimensions,
                                         std::vector<size_t> source_dimensions,
                                         dnnPrimitive_t *p);
  private:
    float negative_slope_;
};
#endif // RELU_LAYER_H
