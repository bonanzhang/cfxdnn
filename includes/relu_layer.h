#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "layer.h"
class ReLULayer : public Layer {
  public:
    virtual void initialize(std::vector<size_t*>);
    virtual void forward();
    virtual void backward(); 
    virtual void update();
    struct input_params {
      float negative_slope = 0.0; 
    };
    ReLULayer(input_params params);   
  private:
    input_params params_; 
    void setForwardInput(std::vector<void*> inputs); 
    void setForwardOutput(std::vector<void*> outputs); 
    void setBackwardInput(std::vector<void*> inputs); 
    void setBackwardOutput(std::vector<void*> outputs);

}; 
#endif // RELU_LAYER_H
