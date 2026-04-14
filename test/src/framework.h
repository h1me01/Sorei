#pragma once

#include <chrono>
#include <cmath>
#include <format>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sorei/nn.h"

namespace sorei::test {

struct TestCase {
    std::string suite;
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& test_registry() {
    static std::vector<TestCase> r;
    return r;
}

struct AutoRegister {
    AutoRegister(const char* suite, const char* name, std::function<void()> fn) {
        test_registry().push_back({suite, name, std::move(fn)});
    }
};

inline int run_all_tests() {
    const int total = test_registry().size();
    int passed = 0, failed = 0;

    println("\n  Sorei Test ({} tests)\n", total);

    std::string cur_suite;
    for (auto& tc : test_registry()) {
        if (tc.suite != cur_suite) {
            if (!cur_suite.empty())
                println("");
            println("  [ {} ]", tc.suite);
            cur_suite = tc.suite;
        }
        sorei::print("    {:<56}", tc.name);
        fflush(stdout);

        auto t0 = std::chrono::steady_clock::now();
        try {
            tc.fn();
            double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
            sorei::println("  PASS  (%.1f ms)", ms);
            ++passed;
        } catch (const std::exception& e) {
            sorei::println("  FAIL");
            sorei::println("        {}", e.what());
            ++failed;
        } catch (...) {
            sorei::println("  FAIL  (unknown exception)");
            ++failed;
        }
    }

    sorei::println("");
    if (failed == 0)
        sorei::println("  All {} tests PASSED", passed);
    else
        sorei::println("  {} / {} PASSED,  {} FAILED", passed, total, failed);
    sorei::println("");

    return failed > 0 ? 1 : 0;
}

// Assertion helpers

[[noreturn]] inline void test_fail(const char* file, int line, const std::string& msg) {
    std::ostringstream os;
    // strip directory prefix for brevity
    const char* slash = file;
    for (const char* p = file; *p; ++p)
        if (*p == '/' || *p == '\\')
            slash = p + 1;
    os << slash << ":" << line << ": " << msg;
    throw std::runtime_error(os.str());
}

} // namespace sorei::test

// Registration macro

#define TEST(suite, name)                                                                          \
    static void _test_fn_##suite##_##name();                                                       \
    static ::sorei::test::AutoRegister _reg_##suite##_##name{                                      \
        #suite, #name, _test_fn_##suite##_##name                                                   \
    };                                                                                             \
    static void _test_fn_##suite##_##name()

// Assertion macros

#define EXPECT_TRUE(cond)                                                                          \
    do {                                                                                           \
        if (!(cond))                                                                               \
            ::sorei::test::test_fail(__FILE__, __LINE__, "Expected true: " #cond);                 \
    } while (0)

#define EXPECT_FALSE(cond)                                                                         \
    do {                                                                                           \
        if (!!(cond))                                                                              \
            ::sorei::test::test_fail(__FILE__, __LINE__, "Expected false: " #cond);                \
    } while (0)

#define EXPECT_EQ(a, b)                                                                            \
    do {                                                                                           \
        auto _a = (a);                                                                             \
        auto _b = (b);                                                                             \
        if (!(_a == _b)) {                                                                         \
            std::ostringstream _oss;                                                               \
            _oss << "Expected equal: " #a " == " #b "\n"                                           \
                 << "          lhs = " << _a << "\n"                                               \
                 << "          rhs = " << _b;                                                      \
            ::sorei::test::test_fail(__FILE__, __LINE__, _oss.str());                              \
        }                                                                                          \
    } while (0)

#define EXPECT_NE(a, b)                                                                            \
    do {                                                                                           \
        auto _a = (a);                                                                             \
        auto _b = (b);                                                                             \
        if (!(_a != _b)) {                                                                         \
            std::ostringstream _oss;                                                               \
            _oss << "Expected not-equal: " #a " != " #b " (both = " << _a << ")";                  \
            ::sorei::test::test_fail(__FILE__, __LINE__, _oss.str());                              \
        }                                                                                          \
    } while (0)

#define EXPECT_NEAR(a, b, tol)                                                                     \
    do {                                                                                           \
        double _a = static_cast<double>(a);                                                        \
        double _b = static_cast<double>(b);                                                        \
        double _d = std::abs(_a - _b);                                                             \
        double _tl = static_cast<double>(tol);                                                     \
        if (_d > _tl) {                                                                            \
            std::ostringstream _oss;                                                               \
            _oss << std::setprecision(6) << "Expected |" #a " - " #b "| <= " #tol " " << "Got |"   \
                 << _a << " - " << _b << "| = " << _d << "  (tol = " << _tl << ")";                \
            ::sorei::test::test_fail(__FILE__, __LINE__, _oss.str());                              \
        }                                                                                          \
    } while (0)

#define EXPECT_LT(a, b)                                                                            \
    do {                                                                                           \
        auto _a = (a);                                                                             \
        auto _b = (b);                                                                             \
        if (!(_a < _b)) {                                                                          \
            std::ostringstream _oss;                                                               \
            _oss << "Expected " #a " < " #b ": " << _a << " >= " << _b;                            \
            ::sorei::test::test_fail(__FILE__, __LINE__, _oss.str());                              \
        }                                                                                          \
    } while (0)

#define EXPECT_GT(a, b)                                                                            \
    do {                                                                                           \
        auto _a = (a);                                                                             \
        auto _b = (b);                                                                             \
        if (!(_a > _b)) {                                                                          \
            std::ostringstream _oss;                                                               \
            _oss << "Expected " #a " > " #b ": " << _a << " <= " << _b;                            \
            ::sorei::test::test_fail(__FILE__, __LINE__, _oss.str());                              \
        }                                                                                          \
    } while (0)

#define EXPECT_LE(a, b)                                                                            \
    do {                                                                                           \
        auto _a = (a);                                                                             \
        auto _b = (b);                                                                             \
        if (!(_a <= _b)) {                                                                         \
            std::ostringstream _oss;                                                               \
            _oss << "Expected " #a " <= " #b ": " << _a << " > " << _b;                            \
            ::sorei::test::test_fail(__FILE__, __LINE__, _oss.str());                              \
        }                                                                                          \
    } while (0)

#define EXPECT_GE(a, b)                                                                            \
    do {                                                                                           \
        auto _a = (a);                                                                             \
        auto _b = (b);                                                                             \
        if (!(_a >= _b)) {                                                                         \
            std::ostringstream _oss;                                                               \
            _oss << "Expected " #a " >= " #b ": " << _a << " < " << _b;                            \
            ::sorei::test::test_fail(__FILE__, __LINE__, _oss.str());                              \
        }                                                                                          \
    } while (0)

#define EXPECT_THROW(expr, ExcType)                                                                \
    do {                                                                                           \
        bool _caught = false;                                                                      \
        try {                                                                                      \
            (void)(expr);                                                                          \
        } catch (const ExcType&) {                                                                 \
            _caught = true;                                                                        \
        } catch (...) {                                                                            \
        }                                                                                          \
        if (!_caught) {                                                                            \
            std::ostringstream _oss;                                                               \
            _oss << "Expected " #ExcType " from: " #expr;                                          \
            ::sorei::test::test_fail(__FILE__, __LINE__, _oss.str());                              \
        }                                                                                          \
    } while (0)

#define EXPECT_GRAD_OK(max_rel, tol)                                                               \
    do {                                                                                           \
        float _mr = (max_rel);                                                                     \
        float _t = (tol);                                                                          \
        if (_mr > _t) {                                                                            \
            std::ostringstream _oss;                                                               \
            _oss << "Gradient check failed: max-relative-error = " << _mr << " (tol = " << _t      \
                 << ")";                                                                           \
            ::sorei::test::test_fail(__FILE__, __LINE__, _oss.str());                              \
        }                                                                                          \
    } while (0)
