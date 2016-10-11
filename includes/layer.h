#ifndef LAYER_H
#define LAYER_H

#include <mkl.h>
#include <stdlib.h>
#include <vector>

class Layer {
  public:
    virtual void forward();
    virtual void backward(); 
    virtual void update();
    virtual void getFwdLayout(dnnLayout_t* playout, dnnResourceType_t type);
    virtual void getBwdLayout(dnnLayout_t* playout, dnnResourceType_t type);
    dnnPrimitive_t forward_p;
    dnnPrimitive_t backward_p;
    void* dnnResources[dnnResourceNumber]; 
};

#endif // LAYER_H
