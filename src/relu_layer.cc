#include "relu_layer.h"
ReLULayer::ReLULayer(float negative_slope) {
  negative_slope_ = negative_slope;
}
void ReLULayer::createForwardPrimitive(std::vector<size_t> previous_destination_dimensions,
                                       std::vector<size_t> destination_dimensions,
                                       dnnPrimitive_t *p) {
    //TODO: fill this in
}
void ReLULayer::createBackwardPrimitive(std::vector<size_t> next_source_dimensions,
                                        std::vector<size_t> source_dimensions,
                                        dnnPrimitive_t *p) {
    //TODO: fill this in
}
