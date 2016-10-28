#ifndef CONVERSION_H
#define CONVERSION_H
#include "mkl_dnn.h"
#include <vector>
class Conversion {
  public:
    Conversion();
    bool needsConversion() const;
    void checkLayouts(dnnPrimitive_t const &primitive,
                      dnnResourceType_t const &resource_type,
                      std::vector<size_t> const &dimensions);
    void convert();
    void setConversionInput(void * const &src);
    void * getConversionOutput() const;
  private:
    bool needs_conversion_;
    dnnPrimitive_t conversion_primitive_;
    void *conversion_input_;
    void *conversion_output_;
};
#endif // CONVERSION_H
