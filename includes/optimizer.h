#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <stdlib.h>
class Optimizer {
  public:
    virtual void applyOptimization(float* weights, float* grad, size_t n, float learning_rate);
};

#endif // OPTIMIZER_H

