cmake_minimum_required(VERSION 2.6)
project(cfxdnn CXX)
set(CFXDNN_LIB_SRCS "${CMAKE_SOURCE_DIR}/src/sequential_network.cc"
                    "${CMAKE_SOURCE_DIR}/src/primitive.cc"
                    "${CMAKE_SOURCE_DIR}/src/conversion.cc"
                    "${CMAKE_SOURCE_DIR}/src/relu_layer.cc"
                    "${CMAKE_SOURCE_DIR}/src/fully_connected_layer.cc"
                    "${CMAKE_SOURCE_DIR}/src/max_pool_layer.cc"
                    "${CMAKE_SOURCE_DIR}/src/avg_pool_layer.cc"
                    "${CMAKE_SOURCE_DIR}/src/convolution_layer.cc"
                    "${CMAKE_SOURCE_DIR}/src/initializer.cc"
                    "${CMAKE_SOURCE_DIR}/src/softmax_objective.cc"
                    "${CMAKE_SOURCE_DIR}/src/sgd.cc"
                    "${CMAKE_SOURCE_DIR}/src/stopwatch.cc")
add_library(cfxdnn SHARED ${CFXDNN_LIB_SRCS})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(BLA_VENDOR Intel10_64lp)
find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories("${CMAKE_SOURCE_DIR}/includes/")
add_executable(runme "${CMAKE_SOURCE_DIR}/main.cc")
target_link_libraries(runme cfxdnn ${BLAS_LIBRARIES})
