#ifndef CONVERSION_H
#define CONVERSION_H
#include "mkl_dnn.h"
#include <vector>
#include <iostream>
class Conversion {
  public:
    Conversion();
    bool needsConversion() const;
    void checkLayouts(dnnPrimitive_t const &primitive,
                      dnnResourceType_t const &resource_type,
                      dnnLayout_t const &actual_layout);
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
