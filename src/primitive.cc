#include "primitive.h"

Primitive::Primitive(vector<size_t> const &input_dims, vector<size_t> &output_dims) {

  const size_t dimention = input_dims.size();
  /* if(dimention != 4) { do something } */

  //TODO: generalize
  // Computing Dimensions. ReLU does not change size. 
  std::copy(input_dims.begin(), input_dims.end(), output_dims.begin());
  
  // Making a copy of input and output dims because the primitive
  // needs size_t*. Also computing the strides for layout
  size_t input_dims_[dimention], output_dims_[dimention];
  for(int i = 0; i < dimention; i++) { 
    input_dims_[i] = input_dims[i]; 
    output_dims_[i] = output_dims[i]; 
  }
  size_t input_strides_[4] = {1,
                              input_dims_[0],
                              input_dims_[0]*input_dims_[1],
                              input_dims_[0]*input_dims_[1]*input_dims_[2]};
  size_t output_strides_[4] = {1,
                              output_dims_[0],
                              output_dims_[0]*output_dims_[1],
                              output_dims_[0]*output_dims_[1]*output_dims_[2]};

  // Creating the forward primitive. 
  dnnLayout_t src_layout;
  dnnLayoutCreate_F32(&src_layout, dimention, input_dims_, input_strides_);
  //TODO: call layer specific stuff
  dnnReLUCreateForward_F32(&forward_p, NULL, src_layout, params_.negative_slope);

  // Creating the backward primitive.
  dnnLayout_t diffdst_layout;
  dnnLayoutCreate_F32(&diffdst_layout, dimention, output_dims_, output_strides_);
  dnnReLUCreateBackward_F32(&backward_p, NULL, diffdst_layout, src_layout, params_.negative_slope);

  // 
  dnnLayoutDelete_F32(diffdst_layout);
  dnnLayoutDelete_F32(src_layout);

}

Primitive::~Primitive() {
  if(forward_p){dnnDelete_F32(forward_p);}
  if(backward_p){dnnDelete_F32(backward_p);}
  if(dst_layout){dnnLayoutDelete_F32(dst_layout);}
  if(diffdst_layout){dnnLayoutDelete_F32(diffdst_layout);}
  if(diffsrc_layout){dnnLayoutDelete_F32(diffsrc_layout);}
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
void Primitive::setFwdInput(void* prev_dst) {
}
void Primitive::setBwdInput(void* next_src) {
}

void* ReLULayer::getResource(dnnResourceType_t type) {
  return layer_resources[type];
}
