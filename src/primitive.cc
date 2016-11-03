#include "primitive.h"
vector<dnnResourceType_t> const Primitive::resource_types{
    dnnResourceFilter,      /* 2  */
    dnnResourceBias,        /* 3  */
    dnnResourceDiffFilter,  /* 5  */
    dnnResourceDiffBias,    /* 6  */
    dnnResourceWorkspace,   /* 8  */
    dnnResourceMultipleSrc, /* 16 */
    dnnResourceMultipleDst  /* 32 */
};
Primitive::Primitive(Layer *layer, vector<size_t> const &src_dimensions)
    : input_dimensions_(src_dimensions),
      forward_primitives_(layer->getNumberOfFwdPrimitives()),
      backward_primitives_(layer->getNumberOfBwdPrimitives()) {
//  std::cout << "constructor for " << layer->getDebugString() << std::endl;
  // Every resource starts as a nullptr, gets filled as required by primitives
  for (int i = 0; i < dnnResourceNumber; i++) {
    resources_[i] = nullptr;
  }
  layer->createPrimitives(input_dimensions_, output_dimensions_,
                          forward_primitives_, backward_primitives_);
  allocateResourcesForPrimitives(forward_primitives_);
  allocateResourcesForPrimitives(backward_primitives_);
  component_name = layer->getDebugString();
}
Primitive::~Primitive() {
  // delete forward primitives
  for (int i = 0; i < forward_primitives_.size(); i++) {
    dnnError_t e = dnnDelete_F32(forward_primitives_[i]);
//    std::cout << "Delete forward primitive " << i
//              << " completed with status: " << e << std::endl;
  }
  // delete backward primitives
  for (int i = 0; i < backward_primitives_.size(); i++) {
    dnnError_t e = dnnDelete_F32(backward_primitives_[i]);
//    std::cout << "Delete backward primitive " << i
//              << " completed with status: " << e << std::endl;
  }
  // delete resource buffers
  for (int i = 0; i < resource_types.size(); i++) {
    if (resources_[resource_types[i]] != nullptr) {
      //      std::cout << "Deleting resource " << resource_types[i];
      dnnError_t e = dnnReleaseBuffer_F32(resources_[resource_types[i]]);
      //      std::cout << " completed with status: " << e << std::endl;
    }
  }
}
void Primitive::forward() {
  if (forward_conversion_.needsConversion()) {
    forward_conversion_.convert();
  }
  for (int i = 0; i < forward_primitives_.size(); i++) {
//    std::cout << "execute forward with input: "
//              << resources_[dnnResourceSrc]
//              << std::endl << std::flush;
//    std::cout << "execute forward with output: "
//              << resources_[dnnResourceDst]
//              << std::endl << std::flush;
    dnnError_t e = dnnExecute_F32(forward_primitives_[i], resources_);
//    std::cout << "forward executed with status: " << e << std::endl;
  }
}
void Primitive::backward() {
  if (backward_conversion_.needsConversion()) {
    std::cout << "backward needs conversion" << std::endl;
    backward_conversion_.convert();
  }
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
    opt.applyOptimization(
        static_cast<float *>(resources_[dnnResourceFilter]),
        static_cast<float *>(resources_[dnnResourceDiffFilter]),
        resource_sizes_[dnnResourceFilter], learning_rate);
  }
  if (resources_[dnnResourceBias] != nullptr &&
      resources_[dnnResourceDiffBias] != nullptr) {
//    std::cout << "updating bias" << std::endl << std::flush;
    opt.applyOptimization(static_cast<float *>(resources_[dnnResourceBias]),
                          static_cast<float *>(resources_[dnnResourceDiffBias]),
                          resource_sizes_[dnnResourceBias], learning_rate);
  }
}
void Primitive::initialize(Initializer const &ini) {
  if (resources_[dnnResourceFilter]) {
    ini.fill(static_cast<float *>(resources_[dnnResourceFilter]),
             resource_sizes_[dnnResourceFilter]);
  }
  if (resources_[dnnResourceBias]) {
    ini.fill(static_cast<float *>(resources_[dnnResourceBias]),
             resource_sizes_[dnnResourceBias]);
  }
}
void Primitive::initializeConversions() {
//  std::cout << "checking forward resources for needed conversions" <<
//  std::endl;
  for (int i = 0; i < forward_primitives_.size(); i++) {
    forward_conversion_.checkLayouts(forward_primitives_[i], dnnResourceSrc,
                                     input_dimensions_);
  }
//  std::cout << "checking backward resources for needed conversions" <<
//  std::endl;
  for (int i = 0; i < backward_primitives_.size(); i++) {
    backward_conversion_.checkLayouts(backward_primitives_[i],
                                      dnnResourceDiffDst, output_dimensions_);
  }
}
void Primitive::setFwdInput(void *src) {
  if (forward_conversion_.needsConversion()) {
    forward_conversion_.setConversionInput(src);
    resources_[dnnResourceSrc] = forward_conversion_.getConversionOutput();
  } else {
    resources_[dnnResourceSrc] = src;
  }
}
void Primitive::setFwdOutput(void *dst) { resources_[dnnResourceDst] = dst; }
void Primitive::setBwdInput(void *diffdst) {
  if (backward_conversion_.needsConversion()) {
    backward_conversion_.setConversionInput(diffdst);
    resources_[dnnResourceDiffDst] = backward_conversion_.getConversionOutput();
  } else {
    resources_[dnnResourceDiffDst] = diffdst;
  }
}
void Primitive::setBwdOutput(void *diffsrc) {
  resources_[dnnResourceDiffSrc] = diffsrc;
}
void Primitive::allocateResourcesForPrimitives(
    vector<dnnPrimitive_t> const &primitives) {
  // for each involved primitive, allocate all the resources it wants
  // in this case, it wants a resource if I can create layout from primitive
  // multiple primitives might want the same resource, do not allocate twice
  for (int i = 0; i < primitives.size(); i++) {
    dnnError_t e;
    dnnLayout_t layout;
    for (int j = 0; j < resource_types.size(); j++) {
      if (resources_[resource_types[j]] == nullptr) {
        e = dnnLayoutCreateFromPrimitive_F32(&layout, primitives[i],
                                             resource_types[j]);
        if (e == E_SUCCESS) {
//          std::cout << "allocating resource type: " <<
//          resource_types[j] << std::endl;
          dnnAllocateBuffer_F32(&resources_[resource_types[j]], layout);
          resource_sizes_[resource_types[j]] =
              dnnLayoutGetMemorySize_F32(layout) / sizeof(float);
          dnnLayoutDelete_F32(layout);
        }
      }
    }
  }
}
std::string Primitive::getComponentName() { return component_name; }
vector<size_t> Primitive::getOutputDimensions() const {
  return output_dimensions_;
}
dnnLayout_t Primitive::getForwardOutputLayout() const {
  dnnLayout_t layout = nullptr;
  dnnError_t e;
  for (int i = 0; i < forward_primitives_.size(); i++) {
    e = dnnLayoutCreateFromPrimitive_F32(&layout, forward_primitives_[i], dnnResourceDst);
    if (e == E_SUCCESS) {
      break;
    }
  }
  return layout;
}
dnnLayout_t Primitive::getBackwardOutputLayout() const {
  dnnLayout_t layout = nullptr;
  dnnError_t e;
  for (int i = 0; i < backward_primitives_.size(); i++) {
    dnnLayoutCreateFromPrimitive_F32(&layout, backward_primitives_[i], dnnResourceDiffSrc);
    if (e == E_SUCCESS) {
      break;
    }
  }
  return layout;
}
