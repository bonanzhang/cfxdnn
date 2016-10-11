#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <mkl.h>
#include <omp.h>

class Layer{
    dnnPrimitive* prims;
    void* dnnRsrc[dnnResourceNumber];
  public:
    virtual void setInput(float* input) {}
    virtual void fwd() {}
    virtual void bkd() {}
};

class Conv {
  public:
    Conv() : public Layer() {

    }
}

class ConvNet {
    std::vector<layer> layers;
  public:
    void add(layer const &l) {
      layers.push_back(l)
    }

    void forward(float* input, float* output) {
      layers[0].dnnRsrc[dnnResourceSrc] = input;
      for(int i = 0; i < layers.size()-1; i++) {
        layers[i].fwd();
　　　　layers[i+1].dnnRsrc[dnnResourceSrc] = 
      }
    }
};


dnnPrimitive_t createConvFwd(const size_t batchSize, const size_t iw, const size_t ih, const size_t ic, const size_t ow, const size_t oh, const size_t oc, const size_t kw, const size_t kh, const size_t stridew, const size_t strideh, const size_t padw, const size_t padh) {

  dnnPrimitive_t convFwd;
  const size_t dim = 4;
  size_t input_dims[dim] = {iw, ih, ic, batchSize};
  size_t input_strides[dim] = {1, iw, iw*ih, iw*ih*ic};
  size_t output_dims[dim] = {ow, oh, oc, batchSize};
  size_t output_strides[dim] = {1, ow, ow*oh, ow*oh*oc};
  size_t filter_dims[dim] = {kw, kh, ic, oc};
  size_t filter_strides[dim] = {1, kw, kw*kh, kw*kh*ic};
  size_t conv_strides[2] = {stridew, strideh};
  int padding[2] = {padw, padh};
  dnnConvolutionCreateForward_F32(&convFwd, 
                                  NULL,
                                  dnnAlgorithmConvolutionDirect,
                                  dim,
                                  input_dims,
                                  output_dims,
                                  filter_dims,
                                  conv_strides,
                                  padding,
                                  dnnBorderZeros
                                  );

  return convFwd;
}


int main() {
  float* input = (float*) malloc(sizeof(float)*64*3*224*224);
  float* output = (float*) malloc(sizeof(float)*64*64*224*224);
  float* filter = (float*) malloc(sizeof(float)*3*3*3*64);
  std::cout << "allocated" << std::endl;

  dnnPrimitive_t pConvFwd = createConvFwd(64,         // batchSize
                                          224,224,3,  // input dims: iw, ih, ic 
                                          224,224,64, // output dims: ow, oh, oc
                                          3,3,        // filter size: kw kh
                                          1,1,        // strides: stridew, strideh
                                          1,1);       // padding: padw, padh

  std::cout << "Created Primitive" << std::endl;
  void* conv_res[dnnResourceNumber];
  conv_res[dnnResourceSrc] = (void*) input;
  conv_res[dnnResourceFilter] = (void*) filter;
  conv_res[dnnResourceDst] = (void*) output;
  for(int i = 0; i < 10; i++) {
    const double t0 = omp_get_wtime();
    dnnExecute_F32(pConvFwd, conv_res);
    const double t1 = omp_get_wtime();
  
    std::cout << ((float*)conv_res[dnnResourceDst])[0] << " time: " << t1-t0 << std::endl;
  }
  dnnDelete_F32(pConvFwd);
  free(input);
  free(output);
  free(filter);
}


