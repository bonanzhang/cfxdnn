CXX=icpc
CXX_FLAGS=-std=c++11
INC_DIR=includes
OBJECTS=main.o fully_connected_layer.o relu_layer.o initializer.o sgd.o softmax_objective.o sequential_network.o primitive.o

#.SUFFIXES: .o .cc

all: $(OBJECTS) runme 

runme : $(OBJECTS) 
	icpc -mkl -o runme $(OBJECTS) 

#.cc.o:
#	$(info )
#	$(info Compiling a CPU object file:)
#	$(CXX) -I$(INC_DIR) -c $(CXXFLAGS) -o "$@" "$<"

main.o : main.cc
	$(CXX) $(CXX_FLAGS) -c main.cc -I$(INC_DIR)

fully_connected_layer.o : src/fully_connected_layer.cc $(INC_DIR)/fully_connected_layer.h
	$(CXX) $(CXX_FLAGS) -c -I$(INC_DIR) -mkl src/fully_connected_layer.cc

initializer.o : src/initializer.cc $(INC_DIR)/initializer.h
	$(CXX) $(CXX_FLAGS) -c -I$(INC_DIR) -mkl src/initializer.cc

primitive.o : src/primitive.cc $(INC_DIR)/primitive.h
	$(CXX) $(CXX_FLAGS) -c -I$(INC_DIR) -mkl src/primitive.cc

relu_layer.o : src/relu_layer.cc $(INC_DIR)/relu_layer.h
	$(CXX) $(CXX_FLAGS) -c -I$(INC_DIR) -mkl src/relu_layer.cc

sequential_network.o : src/sequential_network.cc $(INC_DIR)/sequential_network.h
	$(CXX) $(CXX_FLAGS) -c -I$(INC_DIR) -mkl src/sequential_network.cc

sgd.o : src/sgd.cc $(INC_DIR)/sgd.h
	$(CXX) $(CXX_FLAGS) -c -I$(INC_DIR) -mkl -qopenmp src/sgd.cc

softmax_objective.o : src/softmax_objective.cc $(INC_DIR)/softmax_objective.h
	$(CXX) -c -I$(INC_DIR) -mkl -qopenmp src/softmax_objective.cc

.PHONY: clean
clean:
	rm *.o runme
