
add_executable(astra_example main.cu)
target_link_libraries(astra_example PRIVATE sorei)
target_include_directories(astra_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(astra_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
