cmake_minimum_required(VERSION 3.18)
project(mnist_example LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES native)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. sorei)

add_executable(mnist_example main.cu)
target_link_libraries(mnist_example PRIVATE sorei)
target_include_directories(mnist_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(mnist_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
