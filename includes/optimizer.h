#ifndef OPTIMIZER_H
#define OPTIMIZER_H

class Optimizer {
  public:
    virtual void applyOptimization(float* weights, float* grad, size_t n);
};

#endif // OPTIMIZER_H

