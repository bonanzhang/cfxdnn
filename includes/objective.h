#ifndef OBJECTIVE_H
#define OBJECTIVE_H
#include <stdlib.h>
#include <vector>
class Objective {
    virtual float computeLossAndGradient(size_t const batch_size, size_t const n_classes, float const *src, std::vector<size_t> const &truth, float *diffsrc) = 0; 
};

#endif // OBJECTIVE_H

