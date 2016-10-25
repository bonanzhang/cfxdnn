#ifndef OBJECTIVE_H
#define OBJECTIVE_H
#include <vector>
#include <cstdlib>
class Objective {
    virtual float computeLossAndGradient(size_t const batch_size, size_t const n_classes, float const *src, std::vector<size_t> const &truth, float *diffsrc) const = 0; 
};

#endif // OBJECTIVE_H

