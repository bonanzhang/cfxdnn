#include "primitive.h"
#include <iostream>
Primitive::Primitive(Layer *l, 
                     vector<size_t> const &src_dimensions,
                     vector<size_t> &dst_dimensions)
    : forward_primitives_(l->getNumberOfFwdPrimitives()),
      backward_primitives_(l->getNumberOfBwdPrimitives()),
      requested_fwd_resources_(l->getNumberOfFwdPrimitives()),
      requested_bwd_resources_(l->getNumberOfBwdPrimitives()) {
  // Initializing resource pointers to null (needed for the update() to work)
  for (int i = 0; i < dnnResourceNumber; i++) {
    resources_[i] = nullptr;
  }
  // Vector containing resouce types that are requested by the layer
  l->createPrimitives(src_dimensions, dst_dimensions, forward_primitives_,
                      backward_primitives_, requested_fwd_resources_,
                      requested_bwd_resources_);
  // Allocating requested resources
  for (int i = 0; i < requested_fwd_resources_.size(); i++) {
    for (int j = 0; j < requested_fwd_resources_[i].size(); j++) {
      dnnLayout_t pLayout;
      dnnError_t e = dnnLayoutCreateFromPrimitive_F32(
          &pLayout, forward_primitives_[i], requested_fwd_resources_[i][j]);
      if (e != E_SUCCESS)
        std::cout << "resource layout from primitive failed\n" << std::flush;
      // TODO: errors when layer does not work;
      if (e == E_SUCCESS) {
        dnnAllocateBuffer_F32(&resources_[requested_fwd_resources_[i][j]],
                              pLayout);
        resource_sizes_[requested_fwd_resources_[i][j]] =
            dnnLayoutGetMemorySize_F32(pLayout) / sizeof(float);
        dnnLayoutDelete_F32(pLayout);
      } // else {TODO}
    }
  }
  for (int i = 0; i < requested_bwd_resources_.size(); i++) {
    for (int j = 0; j < requested_bwd_resources_[i].size(); j++) {
      dnnLayout_t pLayout;
      dnnError_t e = dnnLayoutCreateFromPrimitive_F32(
          &pLayout, backward_primitives_[i], requested_bwd_resources_[i][j]);
      if (e == E_SUCCESS) {
        dnnAllocateBuffer_F32(&resources_[requested_bwd_resources_[i][j]],
                              pLayout);
        resource_sizes_[requested_bwd_resources_[i][j]] =
            dnnLayoutGetMemorySize_F32(pLayout) / sizeof(float);
        dnnLayoutDelete_F32(pLayout);
      } // else {TODO}
    }
  }
}

Primitive::~Primitive() {
  // Release Forward resources
  for (int i = 0; i < forward_primitives_.size(); i++) {
    dnnDelete_F32(forward_primitives_[i]);
    for (int j = 0; j < requested_fwd_resources_[i].size(); j++) {
      dnnReleaseBuffer_F32(resources_[requested_fwd_resources_[i][j]]);
    }
  }
  // Release Backward resources
  for (int i = 0; i < backward_primitives_.size(); i++) {
    dnnDelete_F32(backward_primitives_[i]);
    for (int j = 0; j < requested_bwd_resources_[i].size(); j++) {
      dnnReleaseBuffer_F32(resources_[requested_bwd_resources_[i][j]]);
    }
  }
}
void Primitive::forward() {
  for (int i = 0; i < forward_primitives_.size(); i++) {
    std::cout << "execute forward with input: " 
              << resources_[dnnResourceSrc]
              << std::endl << std::flush;
    std::cout << "execute forward with output: "
              << resources_[dnnResourceDst] 
              << std::endl << std::flush;
    dnnExecute_F32(forward_primitives_[i], resources_);
    std::cout << "executed" << std::endl;
  }
}
void Primitive::backward() {
  for (int i = 0; i < backward_primitives_.size(); i++) {
    std::cout << "execute backward with input: " 
              << resources_[dnnResourceDiffDst] 
              << std::endl << std::flush;
    std::cout << "execute backward with output: " 
              << resources_[dnnResourceDiffSrc]
              << std::endl << std::flush;
    dnnExecute_F32(backward_primitives_[i], resources_);
    std::cout << "executed" << std::endl;
  }
}
void Primitive::update(Optimizer *opt, float learning_rate) {
  std::cout << "updating..." << std::flush;
  if (resources_[dnnResourceFilter] != nullptr &&
      resources_[dnnResourceDiffFilter] != nullptr) {
    std::cout << "filter" << std::flush;
    opt->applyOptimization((float *)resources_[dnnResourceFilter],
                           (float *)resources_[dnnResourceDiffFilter],
                           resource_sizes_[dnnResourceFilter], learning_rate);
  }
  std::cout << "..." << std::flush;
  if (resources_[dnnResourceBias] != nullptr &&
      resources_[dnnResourceDiffBias] != nullptr) {
    std::cout << "bias" << std::flush;
    opt->applyOptimization((float *)resources_[dnnResourceBias],
                           (float *)resources_[dnnResourceDiffBias],
                           resource_sizes_[dnnResourceBias], learning_rate);
  }
}
void Primitive::initialize(Initializer *ini) {
  if (resources_[dnnResourceFilter]) {
    ini->fill((float *)resources_[dnnResourceFilter],
              resource_sizes_[dnnResourceFilter]);
  }
  if (resources_[dnnResourceBias]) {
    ini->fill((float *)resources_[dnnResourceBias],
              resource_sizes_[dnnResourceBias]);
  }
}
void Primitive::setFwdInput(void *src) { resources_[dnnResourceSrc] = src; }
void Primitive::setFwdOutput(void *dst) { resources_[dnnResourceDst] = dst; }
void Primitive::setBwdInput(void *diffdst) {
  resources_[dnnResourceDiffDst] = diffdst;
}
void Primitive::setBwdOutput(void *diffsrc) {
  resources_[dnnResourceDiffSrc] = diffsrc;
}
void *Primitive::getResource(dnnResourceType_t type) {
  return resources_[type];
}
