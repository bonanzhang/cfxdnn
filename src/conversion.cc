#include "conversion.h"
Conversion::Conversion()
    : needs_conversion_(false), conversion_primitive_(nullptr),
      conversion_input_(nullptr), conversion_output_(nullptr) {}
bool Conversion::needsConversion() const { return needs_conversion_; }
void Conversion::checkLayouts(dnnPrimitive_t const &primitive,
                              dnnResourceType_t const &resource_type,
                              dnnLayout_t const &actual_layout) {
  if (needs_conversion_) {
    return;
  }
  dnnLayout_t expected_layout;
  dnnError_t e = dnnLayoutCreateFromPrimitive_F32(&expected_layout, primitive,
                                                  resource_type);
  if (e != E_SUCCESS) {
    return;
  }
  if (!dnnLayoutCompare_F32(actual_layout, expected_layout)) {
    dnnConversionCreate_F32(&conversion_primitive_, actual_layout,
                            expected_layout);
    dnnAllocateBuffer_F32(&conversion_output_, expected_layout);
    needs_conversion_ = true;
//    std::cout << "actual buffer size: " << str << std::endl;
//    std::cout << "expected buffer size: "
//              << dnnLayoutGetMemorySize_F32(expected_layout)/4
//              << std::endl;
  }
  dnnLayoutDelete_F32(expected_layout);
}
void Conversion::convert() {
  dnnConversionExecute_F32(conversion_primitive_, conversion_input_,
                           conversion_output_);
}
void Conversion::setConversionInput(void *const &src) {
  conversion_input_ = src;
}
void *Conversion::getConversionOutput() const { return conversion_output_; }
