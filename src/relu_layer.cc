#include "relu_layer.h"

ReLULayer::ReLULayer(ReLULayer::input_params params) {
  params_ = params;
}


void ReLULayer::initFwd(Layer* prev) {
  // Creating the forward primitive. The layout for the src (input for this layer) is 
  // the layout of the previous layer's dst (output of prev)
  prev->getFwdLayout(&src_layout, dnnResourceDst);
  dnnReLUCreateForward_F32(&forward_p, NULL, src_layout, params_.negative_slope);

  // Allocating space for dst
  getFwdLayout(&dst_layout, dnnResourceDst);
  dnnAllocateBuffer_F32((void**)&dnnResources[dnnResourceDst], dst_layout);
}

void ReLULayer::initBwd(Layer* next) {
  // Creating the backward primitive. The layout for the diffdst (input of this layer) 
  // is the layout of the previous layer's diffsrc (output of prev)
  next->getBwdLayout(&diffdst_layout, dnnResourceDiffSrc);
  dnnReLUCreateBackward_F32(&backward_p, NULL, diffdst_layout, src_layout, params_.negative_slope);

  // Allocating space for diffsrc
  getBwdLayout(&diffsrc_layout, dnnResourceDiffSrc);
  dnnAllocateBuffer_F32((void**)&dnnResources[dnnResourceDiffSrc], diffsrc_layout);
}

ReLULayer::~ReLULayer() {
  if(forward_p){dnnDelete_F32(forward_p);}
  if(backward_p){dnnDelete_F32(backward_p);}
  if(src_layout){dnnLayoutDelete_F32(src_layout);}
  if(dst_layout){dnnLayoutDelete_F32(dst_layout);}
  if(diffdst_layout){dnnLayoutDelete_F32(diffdst_layout);}
  if(diffsrc_layout){dnnLayoutDelete_F32(diffsrc_layout);}
  if(dnnResources[dnnResourceDiffSrc]){dnnReleaseBuffer_F32(dnnResources[dnnResourceDiffSrc]);}
  if(dnnResources[dnnResourceDiffDst]){dnnReleaseBuffer_F32(dnnResources[dnnResourceDiffDst]);}
}

void ReLULayer::forward() {
  dnnExecute_F32(forward_p, dnnResources);
}

void ReLULayer::backward() {
  dnnExecute_F32(backward_p, dnnResources);
}

void ReLULayer::update() {}

void ReLULayer::getFwdLayout(dnnLayout_t* layout, dnnResourceType_t type) {
  dnnLayoutCreateFromPrimitive_F32(layout, forward_p, type);
}

void ReLULayer::getBwdLayout(dnnLayout_t* layout, dnnResourceType_t type) {
  dnnLayoutCreateFromPrimitive_F32(layout, backward_p, type);
}

