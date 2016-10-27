#include "primitive.h"
#include <iostream>
Primitive::Primitive(Layer *layer, 
                     vector<size_t> const &src_dimensions) 
    : input_dimensions_(src_dimensions),
      forward_primitives_(layer->getNumberOfFwdPrimitives()),
      backward_primitives_(layer->getNumberOfBwdPrimitives()),
      needs_conversion_(false),
      conversion_primitive_(nullptr),
      conversion_input_(nullptr),
      conversion_output_(nullptr) {
  // Every resource starts as a nullptr, gets filled as required by primitives
  for (int i = 0; i < dnnResourceNumber; i++) {
    resources_[i] = nullptr;
  }
  layer->createPrimitives(input_dimensions_, output_dimensions_, 
                          forward_primitives_, backward_primitives_);
  dnnLayout_t expected_input_layout;
  dnnLayoutCreateFromPrimitive_F32(&expected_input_layout, forward_primitives_[0], dnnResourceSrc);
  size_t const dim = input_dimensions_.size();
  size_t sizes[dim];
  size_t strides[dim];
  size_t str = 1;
  for (int d = 0; d < dim; d++) {
    sizes[d] = input_dimensions_[d];
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
  }
  allocateResourcesForPrimitives(forward_primitives_);
  allocateResourcesForPrimitives(backward_primitives_);
  component_name = layer->getDebugString();
}

Primitive::~Primitive() {
  // delete forward primitives
  for (int i = 0; i < forward_primitives_.size(); i++) {
    dnnError_t e = dnnDelete_F32(forward_primitives_[i]);
    std::cout << "Delete forward primitive " << i << " " << e << std::endl;
  }
  // delete backward primitives
  for (int i = 0; i < backward_primitives_.size(); i++) {
    dnnError_t e = dnnDelete_F32(backward_primitives_[i]);
    std::cout << "Delete backward primitive " << i << " " << e << std::endl;
  }
  // delete resource buffers
  vector<dnnResourceType_t> resource_types{dnnResourceFilter,     /* 2  */
                                           dnnResourceBias,       /* 3  */
                                           dnnResourceDiffFilter, /* 5  */
                                           dnnResourceDiffBias,   /* 6  */
                                           dnnResourceWorkspace,  /* 8  */
                                           dnnResourceMultipleSrc,/* 16 */
                                           dnnResourceMultipleDst /* 32 */
                                           };
  for (int i = 0; i < resource_types.size(); i++) {
    if (resources_[resource_types[i]] != nullptr) {
      dnnError_t e = dnnReleaseBuffer_F32(resources_[resource_types[i]]);
      std::cout << "Delete resource " << resource_types[i] << " " << e << std::endl;
    }
  }
}
void Primitive::forward() {
  if (needs_conversion_) {
    dnnConversionExecute_F32(conversion_primitive_, conversion_input_, conversion_output_);
  }
  for (int i = 0; i < forward_primitives_.size(); i++) {
//    std::cout << "execute forward with input: " 
//              << resources_[dnnResourceSrc]
//              << std::endl << std::flush;
//    std::cout << "execute forward with output: "
//              << resources_[dnnResourceDst] 
//              << std::endl << std::flush;
    dnnError_t e = dnnExecute_F32(forward_primitives_[i], resources_);
    std::cout << "forward executed: " << e << std::endl;
  }
}
void Primitive::backward() {
  for (int i = 0; i < backward_primitives_.size(); i++) {
//    std::cout << "execute backward with input: " 
//              << resources_[dnnResourceDiffDst] 
//              << std::endl << std::flush;
//    std::cout << "execute backward with output: " 
//              << resources_[dnnResourceDiffSrc]
//              << std::endl << std::flush;
    dnnError_t e = dnnExecute_F32(backward_primitives_[i], resources_);
    std::cout << "backward executed: " << e << std::endl;
  }
}
void Primitive::update(Optimizer const &opt, float learning_rate) {
  if (resources_[dnnResourceFilter] != nullptr &&
      resources_[dnnResourceDiffFilter] != nullptr) {
//    std::cout << "updating filter" << std::endl << std::flush;
    opt.applyOptimization(static_cast<float *>(resources_[dnnResourceFilter]),
                          static_cast<float *>(resources_[dnnResourceDiffFilter]),
                          resource_sizes_[dnnResourceFilter], learning_rate);
  }
  if (resources_[dnnResourceBias] != nullptr &&
      resources_[dnnResourceDiffBias] != nullptr) {
//    std::cout << "updating bias" << std::endl << std::flush;
    opt.applyOptimization(static_cast<float *>(resources_[dnnResourceBias]),
                          static_cast<float *>(resources_[dnnResourceDiffBias]),
                          resource_sizes_[dnnResourceBias], learning_rate);
  }
}
void Primitive::initialize(Initializer const &ini) {
  if (resources_[dnnResourceFilter]) {
    ini.fill(static_cast<float *>(resources_[dnnResourceFilter]),
             resource_sizes_[dnnResourceFilter]);
  }
  if (resources_[dnnResourceBias]) {
    ini.fill(static_cast<float *>(resources_[dnnResourceBias]),
             resource_sizes_[dnnResourceBias]);
  }
}
void Primitive::setFwdInput(void *src) {
  if (needs_conversion_) {
    conversion_input_ = src;
    resources_[dnnResourceSrc] = conversion_output_;
  } else {
    resources_[dnnResourceSrc] = src; 
  }
}
void Primitive::setFwdOutput(void *dst) {
  resources_[dnnResourceDst] = dst; 
}
void Primitive::setBwdInput(void *diffdst) {
  resources_[dnnResourceDiffDst] = diffdst;
}
void Primitive::setBwdOutput(void *diffsrc) {
  resources_[dnnResourceDiffSrc] = diffsrc;
}
void * Primitive::getResource(dnnResourceType_t type) {
  return resources_[type];
}
void Primitive::allocateResourcesForPrimitives(vector<dnnPrimitive_t> const &primitives) {
  //for each involved primitive, allocate all the resources it wants
  //in this case, it wants a resource if I can create layout from primitive
  //multiple primitives might want the same resource, do not allocate twice
  vector<dnnResourceType_t> resource_types{dnnResourceFilter,     /* 2  */
                                           dnnResourceBias,       /* 3  */
                                           dnnResourceDiffFilter, /* 5  */
                                           dnnResourceDiffBias,   /* 6  */
                                           dnnResourceWorkspace,  /* 8  */
                                           dnnResourceMultipleSrc,/* 16 */
                                           dnnResourceMultipleDst /* 32 */
                                           };
  for (int i = 0; i < primitives.size(); i++) {
    dnnError_t e;
    dnnLayout_t layout;
    for (int j = 0; j < resource_types.size(); j++) {
      if (resources_[resource_types[j]] == nullptr) {
        e = dnnLayoutCreateFromPrimitive_F32(&layout, 
                                             primitives[i],
                                             resource_types[j]);
        if (e == E_SUCCESS) {
//          std::cout << "allocating resource type: " << resource_types[j] << std::endl;
          dnnAllocateBuffer_F32(&resources_[resource_types[j]], layout);
          resource_sizes_[resource_types[j]] = dnnLayoutGetMemorySize_F32(layout) / sizeof(float);
          dnnLayoutDelete_F32(layout);
        }
      }
    }
  }
}
std::string Primitive::getComponentName() {
  return component_name;
}
vector<size_t> Primitive::getOutputDimensions() const {
  return output_dimensions_;
}
