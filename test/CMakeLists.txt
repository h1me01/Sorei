cmake_minimum_required(VERSION 3.18)
project(sorei_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES native)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../sorei sorei)

add_executable(sorei_test src/main.cu)
target_link_libraries(sorei_test PRIVATE sorei)
target_include_directories(sorei_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(sorei_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
