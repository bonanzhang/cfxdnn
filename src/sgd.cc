#include <sgd.h>

void SGD::applyOptimization(float* weights, float* grad, size_t n, float learning_rate) {
#pragma omp parallel for 
  for(int i = 0; i < n; i++) 
    weights[i] += learning_rate*grad[i];
}
