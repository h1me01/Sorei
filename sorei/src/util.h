#pragma once

#include <cstdlib>
#include <format>
#include <iostream>
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

template <typename T, typename Base>
T* checked_cast(Base* ptr) {
    SOREI_CHECK(ptr);
    T* typed = dynamic_cast<T*>(ptr);
    SOREI_CHECK(typed);
    return typed;
}

} // namespace sorei
