cmake_minimum_required(VERSION 3.18)
project(astra_example LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES native)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../sorei sorei)

add_executable(astra_example src/main.cu)
target_link_libraries(astra_example PRIVATE sorei)
target_include_directories(astra_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(astra_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
