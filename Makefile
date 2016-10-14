runme : main.o fully_connected_layer.o relu_layer.o initializer.o sgd.o softmax_objective.o sequential_network.o primitive.o
	icpc -mkl -o runme main.o fully_connected_layer.o relu_layer.o initializer.o sgd.o softmax_objective.o sequential_network.o primitive.o

main.o : main.cc
	icpc -c main.cc -Iincludes

fully_connected_layer.o : src/fully_connected_layer.cc includes/fully_connected_layer.h
	icpc -c -Iincludes -mkl src/fully_connected_layer.cc

initializer.o : src/initializer.cc includes/initializer.h
	icpc -c -Iincludes -mkl src/initializer.cc

primitive.o : src/primitive.cc includes/primitive.h
	icpc -c -Iincludes -mkl src/primitive.cc

relu_layer.o : src/relu_layer.cc includes/relu_layer.h
	icpc -c -Iincludes -mkl src/relu_layer.cc

sequential_network.o : src/sequential_network.cc includes/sequential_network.h
	icpc -c -Iincludes -mkl src/sequential_network.cc

sgd.o : src/sgd.cc includes/sgd.h
	icpc -c -Iincludes -mkl -qopenmp src/sgd.cc

softmax_objective.o : src/softmax_objective.cc includes/softmax_objective.h
	icpc -c -Iincludes -mkl -qopenmp src/softmax_objective.cc

.PHONY: clean
clean:
	rm *.o runme
