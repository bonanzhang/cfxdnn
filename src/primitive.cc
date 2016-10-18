#include "primitive.h"
#include <iostream>
Primitive::Primitive(Layer *l, std::vector<size_t> const &src_dimensions, std::vector<size_t> &dst_dimensions) {
  // Initializing resource pointers to null (needed for the update() to work) 
  for(int i = 0; i < dnnResourceNumber; i++) {
    resources[i] = NULL;
  }  
  // Initializing the primitives vector;
  size_t const numberOfFwdPrimitives = l->getNumberOfFwdPrimitives();
  size_t const numberOfBwdPrimitives = l->getNumberOfBwdPrimitives();
  forward_p = std::vector<dnnPrimitive_t>(numberOfFwdPrimitives, dnnPrimitive_t());
  backward_p = std::vector<dnnPrimitive_t>(numberOfBwdPrimitives, dnnPrimitive_t());
  requested_fwd_resources = std::vector<std::vector<dnnResourceType_t>>(numberOfFwdPrimitives, std::vector<dnnResourceType_t>());
  requested_bwd_resources = std::vector<std::vector<dnnResourceType_t>>(numberOfBwdPrimitives, std::vector<dnnResourceType_t>());
  // Vector containing resouce types that are requested by the layer
  l->createPrimitives(src_dimensions, dst_dimensions, 
                      forward_p, backward_p, 
                      requested_fwd_resources, requested_bwd_resources);
  // Allocating requested resources 
  for(int i = 0; i < requested_fwd_resources.size(); i++) {
    for(int j = 0; j < requested_fwd_resources[i].size(); j++) {
      dnnLayout_t pLayout;
      int e = dnnLayoutCreateFromPrimitive_F32(&pLayout, forward_p[i], requested_fwd_resources[i][j]);
      //TODO: errors when layer does not work;
      if(pLayout) { 
        dnnAllocateBuffer_F32(&resources[requested_fwd_resources[i][j]], pLayout);
        resource_sizes[requested_fwd_resources[i][j]] = dnnLayoutGetMemorySize_F32(pLayout)/sizeof(float);
        dnnLayoutDelete_F32(pLayout);
      } // else {TODO}
    } 
  }
  for(int i = 0; i < requested_bwd_resources.size(); i++) {
    for(int j = 0; j < requested_bwd_resources[i].size(); j++) {
      dnnLayout_t pLayout;
      dnnLayoutCreateFromPrimitive_F32(&pLayout, backward_p[i], requested_bwd_resources[i][j]);
      if(pLayout) { 
        dnnAllocateBuffer_F32(&resources[requested_bwd_resources[i][j]], pLayout);
        resource_sizes[requested_bwd_resources[i][j]] = dnnLayoutGetMemorySize_F32(pLayout)/sizeof(float);
        dnnLayoutDelete_F32(pLayout);
      } // else {TODO}
    } 
  }
  

  dnnLayout_t pLayout;
  int e = dnnLayoutCreateFromPrimitive_F32(&pLayout, forward_p[0], dnnResourceSrc);
  size_t size = 1;
  for(int i = 0; i < src_dimensions.size(); i++) {
    size*=src_dimensions[i];
  } 

  dnnLayoutCreateFromPrimitive_F32(&pLayout, forward_p[0], dnnResourceDst);
  size = 1;
  for(int i = 0; i < dst_dimensions.size(); i++) {
    size*=dst_dimensions[i];
  } 

  dnnLayoutCreateFromPrimitive_F32(&pLayout, backward_p[0], dnnResourceDiffSrc);
  size = 1;
  for(int i = 0; i < src_dimensions.size(); i++) {
    size*=src_dimensions[i];
  } 

  dnnLayoutCreateFromPrimitive_F32(&pLayout, backward_p[0], dnnResourceDiffDst);
  size = 1;
  for(int i = 0; i < dst_dimensions.size(); i++) {
    size*=dst_dimensions[i];
  } 
}

Primitive::~Primitive() {
  // Release Forward resources
  for(int i = 0; i < forward_p.size(); i++) {
    dnnDelete_F32(forward_p[i]);
    for(int j = 0; j < requested_fwd_resources[i].size(); j++) {
      dnnReleaseBuffer_F32(resources[requested_fwd_resources[i][j]]);
    }
  }
  // Release Backward resources
  for(int i = 0; i < backward_p.size(); i++) {
    dnnDelete_F32(backward_p[i]);
    for(int j = 0; j < requested_bwd_resources[i].size(); j++) {
      dnnReleaseBuffer_F32(resources[requested_bwd_resources[i][j]]);
    }
  }
}
void Primitive::forward() {
  for(int i = 0; i < forward_p.size(); i++) {
    dnnExecute_F32(forward_p[i], resources);
  }
}
void Primitive::backward() {
  for(int i = 0; i < backward_p.size(); i++) {
    dnnExecute_F32(backward_p[i], resources);
  }
}
void Primitive::update(Optimizer *opt, float learning_rate) {
  if (resources[dnnResourceFilter] && resources[dnnResourceDiffFilter]) {
    opt->applyOptimization((float *) resources[dnnResourceFilter], 
                           (float *) resources[dnnResourceDiffFilter],
                           resource_sizes[dnnResourceFilter],
                           learning_rate);
  }
  if (resources[dnnResourceBias] && resources[dnnResourceDiffBias]) {
    opt->applyOptimization((float *) resources[dnnResourceBias], 
                           (float *) resources[dnnResourceDiffBias],
                           resource_sizes[dnnResourceBias],
                           learning_rate);
  }
}
void Primitive::initialize(Initializer *ini) {
  if (resources[dnnResourceFilter]) {
    ini->fill((float *) resources[dnnResourceFilter], resource_sizes[dnnResourceFilter]);
  }
  if (resources[dnnResourceBias]) {
    ini->fill((float *) resources[dnnResourceBias], resource_sizes[dnnResourceBias]);
  }
}
void Primitive::setFwdInput(void *src) {
  resources[dnnResourceSrc] = src;
}
void Primitive::setFwdOutput(void *dst) {
  resources[dnnResourceDst] = dst;
}
void Primitive::setBwdInput(void *diffdst) {
  resources[dnnResourceDiffDst] = diffdst;
}
void Primitive::setBwdOutput(void *diffsrc) {
  resources[dnnResourceDiffSrc] = diffsrc;
}
void * Primitive::getResource(dnnResourceType_t type) {
  return resources[type];
}
