#include "conversion.h"
Conversion::Conversion()
  : needs_conversion_(false),
    conversion_primitive_(nullptr),
    conversion_input_(nullptr),
    conversion_output_(nullptr) { }
bool Conversion::needsConversion() const {
    return needs_conversion_;
}
void Conversion::checkLayouts(dnnPrimitive_t const &primitive,
                              dnnResourceType_t const &resource_type,
                              std::vector<size_t> const &dimensions) {
    dnnLayout_t expected_input_layout;
    dnnError_t e = dnnLayoutCreateFromPrimitive_F32(&expected_input_layout,
                                                    primitive,
                                                    resource_type);
    if (e != E_SUCCESS) {
        return;
    }
    size_t const dim = dimensions.size();
    size_t sizes[dim];
    size_t strides[dim];
    size_t str = 1;
    for (int d = 0; d < dim; d++) {
        sizes[d] = dimensions[d];
        strides[d] = str;
        str *= sizes[d];
    }
    dnnLayout_t input_layout;
    dnnLayoutCreate_F32(&input_layout, dim, sizes, strides);
    if (!dnnLayoutCompare_F32(input_layout, expected_input_layout)) {
        needs_conversion_ = true;
        dnnConversionCreate_F32(&conversion_primitive_,
                                input_layout,
                                expected_input_layout);
        dnnAllocateBuffer_F32(&conversion_output_, expected_input_layout);
    }
    dnnLayoutDelete_F32(expected_input_layout);
    dnnLayoutDelete_F32(input_layout);
}
void Conversion::convert() {
    dnnConversionExecute_F32(conversion_primitive_,
                             conversion_input_,
                             conversion_output_);
}
void Conversion::setConversionInput(void * const &src) {
    conversion_input_ = src;
}
void * Conversion::getConversionOutput() const {
    return conversion_output_;
}
