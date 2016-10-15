#include <softmax_objective.h>
float SoftMaxObjective::computeLossAndGradient(size_t const batch_size, size_t const n_classes, float const *src, std::vector<size_t> const &truth, float *diffsrc) {
  float loss = 0.0f;
  // Optimization: Not collapsing because realistically there aren't enough 
  // classes to support both vectorization and multi-threading
//#pragma omp parallel for reduction(+: loss)
  for(int i = 0; i < batch_size; i++) {
    float sum = 0.0f, max_val = 0.0f;
#pragma omp simd
    for(int j = 0; j < n_classes; j++) {
      max_val = (max_val > src[i*n_classes*j]) ? max_val : src[i*n_classes*j]; 
    }
#pragma omp simd //reduction(+: sum)
    for(int j = 0; j < n_classes; j++) {
      float const score=expf(src[i*n_classes*j]-max_val); 
      diffsrc[i*n_classes*j] = score;
      sum += score; 
    }
    float norm = 1.0f/sum;
#pragma omp simd
    for(int j = 0; j < n_classes; j++) { 
      diffsrc[i*n_classes*j] *= norm;
    }
    loss -= logf(diffsrc[i*n_classes*truth[i]]);
    diffsrc[i*n_classes*truth[i]] -= 1.0f;
  }
  loss /= n_classes;
  return loss; 
}

