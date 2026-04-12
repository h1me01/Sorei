#pragma once

#include <random>

namespace sorei::rng {

template <typename RNG = std::mt19937_64>
auto& thread_local_rng(typename RNG::result_type seed = RNG::default_seed) {
    static thread_local RNG rng(seed);
    return rng;
}

} // namespace sorei::rng