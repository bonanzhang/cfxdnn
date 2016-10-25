#include <iostream>
#include <vector>
#include <array>
#include <cfxdnn.h>
int main() {
  size_t const batch_size = 3;
  size_t const input_c = 3;
  size_t const input_h = 224;
  size_t const input_w = 224;
  size_t const n_classes = 10;
  std::array<float, batch_size*input_c*input_h*input_w> inputData;
  std::cout << "inputData allocated at " << static_cast<void*>(inputData.data()) << std::endl;
  srand(0);
  for(int i = 0; i < batch_size*input_c*input_h*input_w; i++) {
    inputData[i] = ((float)rand())/((float) RAND_MAX)-0.5;
  }
  std::vector<size_t> ground_truth(batch_size, 0);
  SoftMaxObjective obj;
  SGD sgd;

  SequentialNetwork net(batch_size, input_c, input_h, input_w, n_classes);
  net.add_layer(new ConvolutionLayer(3,3, 1,1, 1,1, 3, false)); 
  net.add_layer(new ReLULayer(0.0f)); 
  net.add_layer(new FullyConnectedLayer(n_classes, false)); 
  std::cout << "Finalizing Layers" << std::endl;
  net.finalize_layers();
  std::cout << "Starting training" << std::endl;
  for(int i = 0; i < 0; i++) {
    std::cout << "training iteration " << i << std::endl;
    net.forward(inputData.data());
    std::cout << "forward pass complete" << std::endl;
    float loss = net.getLoss(obj, ground_truth);
    std::cout << "loss calculation complete" << std::endl;
    std::cout << loss << std::endl;
//    std::cout << "starting backward pass" << std::endl;
//    net.backward();
//    std::cout << "backward pass complete" << std::endl;
//    net.update(sgd, 0.001f);
//    std::cout << "update complete" << std::endl;
  }
  std::cout << "Done" << std::endl;
  return 0;
}
