#ifndef SOFTMAX_OBJECTIVE_H
#define SOFTMAX_OBJECTIVE_H
#include "objective.h"
#include <cmath>
#include <iostream>
class SoftMaxObjective : public Objective {
  public:
    float computeLossAndGradient(size_t batch_size, size_t n_classes, float const *src, std::vector<size_t> const &truth, float *diffsrc) const;
};

#endif // SOFTMAX_OBJECTIVE_H

