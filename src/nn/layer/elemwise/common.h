#pragma once

#include <string>
#include <variant>

namespace sorei::nn::layer {

template <typename Variant>
std::string elemwise_op_name(const Variant& op) {
    return std::string(std::visit([](auto&& o) { return std::decay_t<decltype(o)>::name; }, op));
}

} // namespace sorei::nn::layer
