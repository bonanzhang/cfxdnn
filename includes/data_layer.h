#ifndef DATA_LAYER_H
#define DATA_LAYER_H

#include "layer.h"
class DataLayer : public Layer {
  public:
    void forward();
    void backward(); 
    void update();
    void initFwd(Layer* prev);
    void initBwd(Layer* next);
    void getFwdLayout(dnnLayout_t* playout, dnnResourceType_t type); 
    void getBwdLayout(dnnLayout_t* playout, dnnResourceType_t type);
    struct input_params {
      size_t* dims; 
    };
    DataLayer(input_params* params);  
    ~DataLayer();
    input_params* params_;
  private:
    dnnLayout_t dst_layout; 
}; 
#endif // DATA_LAYER_H
