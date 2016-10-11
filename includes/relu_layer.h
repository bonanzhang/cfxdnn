#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "layer.h"
class ReLULayer : public Layer {
  public:
    void forward();
    void backward(); 
    void update();
    void getFwdLayout(dnnLayout_t* playout, dnnResourceType_t type); 
    void getBwdLayout(dnnLayout_t* playout, dnnResourceType_t type);
    struct input_params {
      float negative_slope = 0.0; 
    };
    ReLULayer(input_params* params, Layer* previous_layer, Layer* next_layer);  
    ~ReLULayer();
    input_params* params_; 
  private:
    dnnLayout_t src_layout;
    dnnLayout_t dst_layout;
    dnnLayout_t diffdst_layout;
    dnnLayout_t diffsrc_layout;
}; 
#endif // RELU_LAYER_H
