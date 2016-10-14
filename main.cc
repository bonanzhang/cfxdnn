#include <stdlib.h>
#include <cfxdnn.h>
int main() {
  size_t const batch_size = 64;
  size_t const input_c= 2;
  size_t const input_h = 224;
  size_t const input_w = 224;
  float *inputData = (float *) malloc(sizeof(float)*batch_size*input_c*input_h*input_w);
  SequentialNetwork net(batch_size, input_c, input_h, input_w);
  net.add_layer(new FullyConnectedLayer(10, false)); 
  net.add_layer(new ReLULayer(0.0f)); 
  net.finalize_layers();
  //net.forward();
}
