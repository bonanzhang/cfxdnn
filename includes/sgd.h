#ifndef SGD_H
#define SGD_H
#include <optimizer.h>
class SGD : public Optimizer {
  public:
    void applyOptimization(float* weights, float* grad, size_t n, float learning_rate);
};

#endif // SGD_H

