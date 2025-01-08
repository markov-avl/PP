#ifndef BIT_SHUFFLE_H
#define BIT_SHUFFLE_H

#include <bit>

template<typename T>
void bit_shuffle(const T *in, T *out, const std::size_t n) {
    const std::size_t log2_n = std::countr_zero(n);

    for (std::size_t i = 0; i < n; ++i) {
        std::size_t reversed_index = 0;
        std::size_t index = i;

        for (std::size_t j = 0; j < log2_n; ++j) {
            reversed_index = reversed_index << 1 | index & 1;
            index >>= 1;
        }

        out[reversed_index] = in[i];
    }
}

#endif
