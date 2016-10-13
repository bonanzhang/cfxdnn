#include "primitive.h"

Primitive::Primitive(Layer* l, vector<size_t> const &src_dimentions, vector<size_t> &dst_dimensions) {
  // TODO? Maybe check for dimension here
  // Vector containing resouce types that are requested by the layer
  l->createPrimitives(src_dimensions, dst_dimensions, forward_p, backward_p, requested_fwd_resources, requested_bwd_resources);

  // Allocating requested resources 
  for(int i = 0; i < requested_fwd_resources.size(); i++) {
    dnnLayout_t pLayout;
    dnnLayoutCreateFromPrimitive(&pLayout, forward_p, requested_fwd_resources[i]);
    if(playout) { 
      dnnAllocateBuffer(resources[i], pLayout);
      dnnLayoutDelete(pLayout);
    } // else {TODO}
  }
  for(int i = 0; i < requested_bwd_resources.size(); i++) {
    dnnLayout_t pLayout;
    dnnLayoutCreateFromPrimitive(&pLayout, backward_p, requested_bwd_resources[i]);
    if(playout) { 
      dnnAllocateBuffer(resources[i], pLayout);
      dnnLayoutDelete(pLayout);
    } // else {TODO}
  }
}

Primitive::~Primitive() {
  // Delete the primitives
  dnnDelete(forward_p);
  dnnDelete(backward_p);
  // Release Forward resources
  for(int i = 0; i < requested_fwd_resources.size(); i++) {
    dnnReleaseBuffer(resources[i]);
  }
  // Release Bacward resources
  for(int i = 0; i < requested_bwd_resources.size(); i++) {
    dnnReleaseBuffer(resources[i]);
  }
  // TODO: Delete vectors?
}

void Primitive::forward() {
  dnnExecute_F32(forward_p, layer_resources);
}

void Primitive::backward() {
  dnnExecute_F32(backward_p, layer_resources);
}

void Primitive::update(Optimizer opt) {
  //TODO: update sometimes, depending on the layer
}

void setFwdInput(void* src) {
}

void setFwdOutput(void* dst) {
}

void setBwdInput(void* diffdst) {
}

void setBwdOutput(void* diffsrc) {
}

void* ReLULayer::getResource(dnnResourceType_t type) {
  return layer_resources[type];
}
