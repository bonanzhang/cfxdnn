cmake_minimum_required(VERSION 2.6)
project(cfxdnn CXX)
set(CFXDNN_LIB_SRCS "${CMAKE_SOURCE_DIR}/src/sequential_network.cc"
                    "${CMAKE_SOURCE_DIR}/src/primitive.cc"
                    "${CMAKE_SOURCE_DIR}/src/relu_layer.cc"
                    "${CMAKE_SOURCE_DIR}/src/fully_connected_layer.cc"
                    "${CMAKE_SOURCE_DIR}/src/initializer.cc"
                    "${CMAKE_SOURCE_DIR}/src/softmax_objective.cc"
                    "${CMAKE_SOURCE_DIR}/src/sgd.cc")
add_library(cfxdnn SHARED ${CFXDNN_LIB_SRCS})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
find_package(MKLRT REQUIRED)
include_directories("${CMAKE_SOURCE_DIR}/includes/"
                    ${MKL_RT_INCLUDE_DIRS})
add_executable(runme "${CMAKE_SOURCE_DIR}/main.cc")
target_link_libraries(runme cfxdnn ${MKL_RT_LIBRARIES})
