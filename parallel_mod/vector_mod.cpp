#pragma once

#include <thread>
#include <vector>
#include <barrier>
#include "mod_ops.h"
#include "num_threads.h"
#include "vector_mod.h"

#ifndef HARDWARE_CONSTRUCTIVE_INTERFERENCE_SIZE
#define HARDWARE_CONSTRUCTIVE_INTERFERENCE_SIZE 64
#endif

struct partial_result_t {
    alignas(HARDWARE_CONSTRUCTIVE_INTERFERENCE_SIZE) IntegerWord value;
};


struct thread_range {
    std::size_t b, e;
};


IntegerWord pow_mod(const IntegerWord base, const IntegerWord power, const IntegerWord mod) {
    IntegerWord result = 1;
    IntegerWord temp_value = base;
    IntegerWord temp_power = power;
    while (temp_power > 0) {
        if (temp_power % 2 != 0) {
            result = mul_mod(result, temp_value, mod);
        }
        temp_power >>= 1;
        temp_value = mul_mod(temp_value, temp_value, mod);
    }
    return result;
}


IntegerWord word_pow_mod(const size_t power, const IntegerWord mod) {
    return pow_mod((-mod) % mod, power, mod);
    // return pow_mod((~mod + 1) % mod, power, mod);
}

thread_range vector_thread_range(const size_t n, const unsigned T, const unsigned t) {
    auto b = n % T;
    auto s = n / T;
    if (t < b) {
        b = ++s * t;
    } else {
        b += s * t;
    }
    const size_t e = b + s;
    return thread_range(b, e);
};

IntegerWord vector_mod(const IntegerWord *V, std::size_t N, IntegerWord mod) {
    size_t T = get_num_threads();
    std::vector<std::thread> threads(T);
    std::vector<partial_result_t> partial_results(T);
    std::barrier bar(T);

    auto thread_lambda = [V, N, T, mod, &partial_results, &bar](const unsigned t) {
        auto [b, e] = vector_thread_range(N, T, t);

        IntegerWord sum = 0;
        // Схема Хорнера
        for (unsigned i = e; b < i;) {
            //sum = (sum * x + V[e-1-i]) % mod;
            sum = add_mod(times_word(sum, mod), V[--i], mod); // то же самое, но без переполнения
        }
        partial_results[t].value = sum;
        for (size_t i = 1, ii = 2; i < T; i = ii, ii += ii) {
            bar.arrive_and_wait();
            if (t % ii == 0 && t + i < T) {
                auto [nb, _] = vector_thread_range(N, T, t + i);
                partial_results[t].value = add_mod(
                    partial_results[t].value,
                    mul_mod(partial_results[t + i].value, word_pow_mod(nb - b, mod), mod),
                    mod
                );
            }
        }
    };

    for (std::size_t i = 0; i < T; ++i) {
        threads[i] = std::thread(thread_lambda, i);
    }

    for (auto &i: threads) {
        i.join();
    }

    return partial_results[0].value;
}
