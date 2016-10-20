#include "initializer.h"
void Initializer::fill(float *v, int n) const {
    // TODO: user defined seed, mean, variance
//    std::random_device rd;
//    std::mt19937 mt(rd);
    std::mt19937 mt(1729);
    std::normal_distribution<float> d(0.0f, 0.01f);
    for (int i = 0; i < n; i++) {
        v[i] = d(mt);
    }
}
