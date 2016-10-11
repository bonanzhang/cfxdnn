#include "relu_layer.h"

void ReLULayer::initialize(std::vector<size_t*> dims) {
  dnnLayout_t pLayout;
  size_t strides[] = {1, dims[0][0], dims[0][0]*dims[0][1],dims[0][0]*dims[0][1]*dims[0][2]};
  dnnLayoutCreate_F32(&pLayout, 2, dims[0], strides);
  dnnReLUCreateForward_F32(&forward_p, NULL, pLayout, 0);
  dnnReLUCreateBackward_F32(&backward_p, NULL, pLayout, pLayout, 0);
}

ReLULayer::ReLULayer(ReLULayer::input_params params) {
  params_ = params;
}

void ReLULayer::forward() {
  dnnExecute_F32(forward_p, dnnResources);
}

void ReLULayer::backward() {
  dnnExecute_F32(backward_p, dnnResources);
}

void ReLULayer::update() {}

void ReLULayer::setForwardInput(std::vector<void*> inputs) {
  dnnResources[dnnResourceSrc] = inputs[0];
}

void ReLULayer::setForwardOutput(std::vector<void*> outputs) {
  dnnResources[dnnResourceDst] = outputs[0];
}

void ReLULayer::setBackwardInput(std::vector<void*> inputs) {
  dnnResources[dnnResourceDiffDst] = inputs[0];
}

void ReLULayer::setBackwardOutput(std::vector<void*> outputs) {
  dnnResources[dnnResourceDiffSrc] = outputs[0];
}

