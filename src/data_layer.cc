#include "data_layer.h"

DataLayer::DataLayer(DataLayer::input_params* params) {
  params_ = params;
}


void DataLayer::initFwd(Layer* prev) {
}

void DataLayer::initBwd(Layer* next) {
}

DataLayer::~DataLayer() {
  free(params_->dims);
  delete params_;
}

void DataLayer::forward() {
}

void DataLayer::backward() {
}

void DataLayer::update() {}

void DataLayer::getFwdLayout(dnnLayout_t* playout, dnnResourceType_t type) {
  if(type == dnnResourceDst) {
    size_t strides[4] = {1, 
                         params_->dims[0], 
                         params_->dims[0]*params_->dims[1], 
                         params_->dims[0]*params_->dims[1]*params_->dims[2]};
    dnnLayoutCreate_F32(playout, 4, params_->dims, strides);
  }
}

void DataLayer::getBwdLayout(dnnLayout_t* playout, dnnResourceType_t type) {
}

