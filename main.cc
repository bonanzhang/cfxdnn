#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cfxdnn.h>
int main() {
  size_t const batch_size = 10;
  size_t const input_c= 3;
  size_t const input_h = 32;
  size_t const input_w = 32;
  size_t const n_classes = 10;
  float *inputData = (float *) malloc(sizeof(float)*batch_size*input_c*input_h*input_w);
  srand(0);
  for(int i = 0; i < batch_size*input_c*input_h*input_w; i++) inputData[i] = ((float)rand())/((float) RAND_MAX)-0.5;
  std::vector<size_t> ground_truth = {0,0,0,1,1,0,1,0,1,1}; 
  SoftMaxObjective obj;
  SGD sgd;

  SequentialNetwork net(batch_size, input_c, input_h, input_w, n_classes);
  net.add_layer(new ConvolutionLayer(5,5, 1,1, 2,2, 32, false)); 
  net.add_layer(new MaxPoolLayer(3,3, 2,2, 1,1)); 
  net.add_layer(new ReLULayer(0.0f)); 

  net.add_layer(new ConvolutionLayer(5,5, 1,1, 2,2, 32, false)); 
  net.add_layer(new ReLULayer(0.0f)); 
  net.add_layer(new AvgPoolLayer(3,3, 2,2, 1,1)); 

  net.add_layer(new ConvolutionLayer(5,5, 1,1, 2,2, 64, false)); 
  net.add_layer(new ReLULayer(0.0f)); 
  net.add_layer(new AvgPoolLayer(3,3, 2,2, 1,1)); 

  net.add_layer(new FullyConnectedLayer(64, false)); 
  net.add_layer(new FullyConnectedLayer(10, false)); 
  std::cout << "Finalizing Layers" << std::endl;
  net.finalize_layers();
  std::cout << "start" << std::endl;
  
  for(int i = 0; i < 5; i++) {
    net.forward(inputData);
    float loss = net.getLoss(&obj, ground_truth);
    std::cout << loss << std::endl;
    //net.backward();
    //inet.update(&sgd, 0.00001f);
  }
  std::cout << "Done" << std::endl;
}
