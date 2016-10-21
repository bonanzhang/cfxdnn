#include "primitive.h"
#include <iostream>
Primitive::Primitive(Layer *layer, 
                     vector<size_t> const &src_dimensions,
                     vector<size_t> &dst_dimensions)
    : forward_primitives_(layer->getNumberOfFwdPrimitives()),
      backward_primitives_(layer->getNumberOfBwdPrimitives()) {
  // Every resource starts as a nullptr, gets filled as required by primitives
  for (int i = 0; i < dnnResourceNumber; i++) {
    resources_[i] = nullptr;
  }
  layer->createPrimitives(src_dimensions, dst_dimensions, 
                      forward_primitives_, backward_primitives_);
  allocateResourcesForPrimitives(forward_primitives_);
  allocateResourcesForPrimitives(backward_primitives_);
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
    opt.applyOptimization((float *)resources_[dnnResourceFilter],
                          (float *)resources_[dnnResourceDiffFilter],
                          resource_sizes_[dnnResourceFilter], learning_rate);
  }
  if (resources_[dnnResourceBias] != nullptr &&
      resources_[dnnResourceDiffBias] != nullptr) {
//    std::cout << "updating bias" << std::endl << std::flush;
    opt.applyOptimization((float *)resources_[dnnResourceBias],
                          (float *)resources_[dnnResourceDiffBias],
                          resource_sizes_[dnnResourceBias], learning_rate);
  }
}
void Primitive::initialize(Initializer const &ini) {
  if (resources_[dnnResourceFilter]) {
    ini.fill((float *)resources_[dnnResourceFilter],
             resource_sizes_[dnnResourceFilter]);
  }
  if (resources_[dnnResourceBias]) {
    ini.fill((float *)resources_[dnnResourceBias],
             resource_sizes_[dnnResourceBias]);
  }
}
void Primitive::setFwdInput(void *src) {
  resources_[dnnResourceSrc] = src; 
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
