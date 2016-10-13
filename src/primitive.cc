#include "primitive.h"

Primitive::Primitive(Layer* l, std::vector<size_t> const &src_dimensions, std::vector<size_t> &dst_dimensions) {
  // TODO? Maybe check for dimension here
  // Vector containing resouce types that are requested by the layer
  l->createPrimitives(src_dimensions, dst_dimensions, &forward_p, &backward_p, requested_fwd_resources, requested_bwd_resources);

  // Initializing resource pointers to null (needed for the update() to work) 
  for(int i = 0; i < dnnResourceNumber; i++) {
    resources[i] = NULL;
  }  
  // Allocating requested resources 
  for(int i = 0; i < requested_fwd_resources.size(); i++) {
    dnnLayout_t pLayout;
    dnnLayoutCreateFromPrimitive_F32(&pLayout, forward_p, requested_fwd_resources[i]);
    if(pLayout) { 
      dnnAllocateBuffer_F32(&resources[i], pLayout);
      resource_sizes[i] = dnnLayoutGetMemorySize_F32(pLayout)/sizeof(float);
      dnnLayoutDelete_F32(pLayout);
    } // else {TODO}
  }
  for(int i = 0; i < requested_bwd_resources.size(); i++) {
    dnnLayout_t pLayout;
    dnnLayoutCreateFromPrimitive_F32(&pLayout, backward_p, requested_bwd_resources[i]);
    if(pLayout) { 
      dnnAllocateBuffer_F32(&resources[i], pLayout);
      resource_sizes[i] = dnnLayoutGetMemorySize_F32(pLayout)/sizeof(float);
      dnnLayoutDelete_F32(pLayout);
    } // else {TODO}
  }
}

Primitive::~Primitive() {
  // Delete the primitives
  dnnDelete_F32(forward_p);
  dnnDelete_F32(backward_p);
  // Release Forward resources
  for(int i = 0; i < requested_fwd_resources.size(); i++) {
    dnnReleaseBuffer_F32(resources[i]);
  }
  // Release Backward resources
  for(int i = 0; i < requested_bwd_resources.size(); i++) {
    dnnReleaseBuffer_F32(resources[i]);
  }
}

void Primitive::forward() {
  dnnExecute_F32(forward_p, resources);
}

void Primitive::backward() {
  dnnExecute_F32(backward_p, resources);
}

void Primitive::update(Optimizer* opt) {
  if(resources[dnnResourceFilter] && resources[dnnResourceDiffFilter]) {
    opt->applyOptimization((float*) resources[dnnResourceFilter], 
                           (float*) resources[dnnResourceDiffFilter],
                           resource_sizes[dnnResourceFilter]);
  }
  if(resources[dnnResourceBias] && resources[dnnResourceDiffBias]) {
    opt->applyOptimization((float*) resources[dnnResourceBias], 
                           (float*) resources[dnnResourceDiffBias],
                           resource_sizes[dnnResourceBias]);
  }
}

void Primitive::setFwdInput(void* src) {
  resources[dnnResourceSrc] = src;
}

void Primitive::setFwdOutput(void* dst) {
  resources[dnnResourceDst] = dst;
}

void Primitive::setBwdInput(void* diffdst) {
  resources[dnnResourceDiffDst] = diffdst;
}

void Primitive::setBwdOutput(void* diffsrc) {
  resources[dnnResourceDiffSrc] = diffsrc;
}

void* Primitive::getResource(dnnResourceType_t type) {
  return resources[type];
}
