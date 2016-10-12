#include "data_layer.h"

DataLayer::DataLayer(DataLayer::input_params params) {
  params_ = params;
}


void DataLayer::initFwd(Layer* prev) {
}

void DataLayer::initBwd(Layer* next) {
}

DataLayer::~DataLayer() {
  delete params_;
}

void DataLayer::forward() {
}

void DataLayer::backward() {
}

void DataLayer::update() {}

void DataLayer::getFwdLayout(dnnLayout_t* playout, dnnResourceType_t type) {
  if(type == dnnResourceDst) {
    size_t dims[4] = {params_.input_w,
                       params_.input_h,
                       params_.input_c,
                       params_.batc_size} 
    size_t strides[4] = {1, 
                         dims[0], 
                         dims[0]*dims[1], 
                         dims[0]*dims[1]*dims[2]};
    dnnLayoutCreate_F32(playout, 4, dims, strides);
  }
}

void DataLayer::getBwdLayout(dnnLayout_t* playout, dnnResourceType_t type) {
}

