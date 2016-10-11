C++ wrapper for the MKLDNN primitives.

Net - Container for layers. The user should only have to call fwd(), bkd() and update() member functions of this class. Currently only sequencial networks. 
  - potentiallly graph and sequential supports
Layer - Base class for various neural network layers. Conatins the weight and gradient.
  - conv 
  - relu
  - maxpool
Optimizer - optimization algorithm object to be passed into net.update().
Loader? - should be able to load a network from some sort of configuration file.
