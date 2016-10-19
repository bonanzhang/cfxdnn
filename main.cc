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
  std::cout << "inputData allocated at " << static_cast<void*>(inputData) << std::endl;
  srand(0);
  for(int i = 0; i < batch_size*input_c*input_h*input_w; i++) {
    inputData[i] = ((float)rand())/((float) RAND_MAX)-0.5;
  }
  std::vector<size_t> ground_truth = {0,0,0,1,1,0,1,0,1,1}; 
  SoftMaxObjective obj;
  SGD sgd;

  SequentialNetwork net(batch_size, input_c, input_h, input_w, n_classes);
  //component 0, 1
  net.add_layer(new ConvolutionLayer(5,5, 1,1, 2,2, 32, false)); 
  //component 2
  net.add_layer(new MaxPoolLayer(3,3, 2,2, 1,1)); 
  //component 3
  net.add_layer(new ReLULayer(0.0f)); 
  //component 4, 5
  net.add_layer(new ConvolutionLayer(5,5, 1,1, 2,2, 32, false)); 
  //component 6
  net.add_layer(new ReLULayer(0.0f)); 
  //component 7
  net.add_layer(new AvgPoolLayer(3,3, 2,2, 1,1)); 
  //component 8, 9
  net.add_layer(new ConvolutionLayer(5,5, 1,1, 2,2, 64, false)); 
  //component 10
  net.add_layer(new ReLULayer(0.0f)); 
  //component 11
  net.add_layer(new AvgPoolLayer(3,3, 2,2, 1,1)); 
  //component 12
  net.add_layer(new FullyConnectedLayer(64, false)); 
  //component 13
  net.add_layer(new FullyConnectedLayer(10, false)); 
  std::cout << "Finalizing Layers" << std::endl;
  net.finalize_layers();
  std::cout << "Starting training" << std::endl;
  
  for(int i = 0; i < 5; i++) {
    std::cout << "training iteration " << i << std::endl;
    net.forward(inputData);
    std::cout << "forward pass complete" << std::endl;
    float loss = net.getLoss(&obj, ground_truth);
    std::cout << "loss calculation complete" << std::endl;
    std::cout << loss << std::endl;
//    std::cout << "starting backward pass" << std::endl;
//    net.backward();
//    std::cout << "backward pass complete" << std::endl;
//    net.update(&sgd, 0.001f);
//    std::cout << "update complete" << std::endl;
  }
  free(inputData);
  std::cout << "Done" << std::endl;
  return 0;
}
