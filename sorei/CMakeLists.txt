cmake_minimum_required(VERSION 3.18)
project(Sorei LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES native)

file(GLOB_RECURSE SOREI_SRCS CONFIGURE_DEPENDS src/*.cu src/*.cpp)
add_library(sorei ${SOREI_SRCS})

target_include_directories(sorei PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(sorei PUBLIC CUDA::cudart CUDA::cublas)

target_compile_options(sorei PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    $<$<CONFIG:Debug>:   -G>
    $<$<CONFIG:Release>: --use_fast_math -lineinfo>
  >
)

set_target_properties(sorei PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
