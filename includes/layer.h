#ifndef LAYER_H
#define LAYER_H

#include <mkl.h>
#include <stdlib.h>
#include <vector>

class Layer {
  public:
    virtual void initialize(std::vector<size_t*>);
    virtual void forward();
    virtual void backward(); 
    virtual void update();
    dnnPrimitive_t forward_p;
    dnnPrimitive_t backward_p;
    void* dnnResources[dnnResourceNumber]; 
  private:
    virtual void setForwardInput(std::vector<void*> inputs); 
    virtual void setForwardOutput(std::vector<void*> outputs); 
    virtual void setBackwardInput(std::vector<void*> inputs); 
    virtual void setBackwardOutput(std::vector<void*> outputs);
};

#endif // LAYER_H
