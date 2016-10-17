#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cfxdnn.h>
int main() {
  size_t const batch_size = 10;
  size_t const input_c= 1;
  size_t const input_h = 10;
  size_t const input_w = 10;
  float *inputData = (float *) malloc(sizeof(float)*batch_size*input_c*input_h*input_w);
  srand(0);
  for(int i = 0; i < batch_size*input_c*input_h*input_w; i++) inputData[i] = ((float)rand())/((float) RAND_MAX)-0.5;
  std::vector<size_t> ground_truth = {0,0,0,1,1,0,1,0,1,1}; 
  SoftMaxObjective obj;
  SGD sgd;

  SequentialNetwork net(batch_size, input_c, input_h, input_w, 2);
  net.add_layer(new FullyConnectedLayer(2, false)); 
  //net.add_layer(new ReLULayer(0.0f)); 
  net.finalize_layers();
  std::cout << "start" << std::endl;
  
  for(int i = 0; i < 10; i++) {
    net.forward(inputData);
    float loss = net.getLoss(&obj, ground_truth);
    std::cout << loss << std::endl;
    net.backward();
    net.update(&sgd, 0.001f);
  }
}
