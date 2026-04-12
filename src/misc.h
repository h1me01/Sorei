#pragma once

#include <cstdlib>
#include <format>
#include <iostream>
#include <string>
#include <string_view>

#define SOREI_CHECK(expr)                                                                          \
    do {                                                                                           \
        if (!static_cast<bool>(expr)) {                                                            \
            printf("CHECK failed: %s\n", #expr);                                                   \
            printf("    file: %s\n", __FILE__);                                                    \
            printf("    line: %d\n", __LINE__);                                                    \
            printf("    func: %s\n", __FUNCTION__);                                                \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

#define SOREI_CUDA_CHECK(expr)                                                                     \
    do {                                                                                           \
        cudaError_t result = (expr);                                                               \
        if (result != cudaSuccess) {                                                               \
            printf("CUDA error: error when calling %s\n", #expr);                                  \
            printf("    file: %s\n", __FILE__);                                                    \
            printf("    line: %d\n", __LINE__);                                                    \
            printf("    error: %s\n", cudaGetErrorString(result));                                 \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

namespace sorei {

template <typename... Args>
inline void print(std::string_view fmt, Args&&... args) {
    std::cout << std::vformat(fmt, std::make_format_args(args...));
}

template <typename... Args>
inline void println(std::string_view fmt, Args&&... args) {
    print(fmt, std::forward<Args>(args)...);
    std::cout << '\n';
}

template <typename... Args>
[[noreturn]] inline void error(std::string_view fmt, Args&&... args) {
    std::cerr << "Error | " << std::vformat(fmt, std::make_format_args(args...)) << '\n';
    std::abort();
}

inline void set_device(int id) { SOREI_CUDA_CHECK(cudaSetDevice(id)); }

inline std::string device_info() {
    int device = -1;
    SOREI_CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop{};
    SOREI_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    return prop.name;
}

} // namespace sorei
