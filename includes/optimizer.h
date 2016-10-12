#ifndef OPTIMIZER_H
#define OPTIMIZER_H

class Optimizer {
  public:
    virtual void applyOptimization(float* weights, float* grad);
};

#endif // OPTIMIZER_H

