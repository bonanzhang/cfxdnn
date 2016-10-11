C++ wrapper for the MKLDNN primitives.

Net - Container for layers. The user should only have to call fwd(), bkd() and update() member functions of this class. Currently only sequencial networks. 
  - potentiallly graph and sequential supports
Layer - Base class for various neural network layers. Conatins the weight and gradient.
  - conv 
  - relu
  - maxpool
Optimizer - optimization algorithm object to be passed into net.update().
Loader? - should be able to load a network from some sort of configuration file.

Tasks a user will probably do:
define a network
come up with some hyper parameters
give it data
press go to train
look at learning curve data
modify the network or hyper parameter
press go to train again
save network weights
save network
go do something else
load network
load network weights
individually run forward
look at array of losses
individually run backward
individually run update
