#include "bit_shuffle.h"
#include "implementations.h"


void transform(Complex *output, const size_t n) {
    const size_t half = n / 2;
    for (std::size_t k = 0; k < half; ++k) {
        auto W = std::polar(1.0, -2.0 * static_cast<double>(k) * std::numbers::pi_v<double> / static_cast<double>(n));
        auto t1 = output[k];
        auto t2 = output[k + half];
        output[k] = t1 + W * t2;
        output[k + half] = t1 - W * t2;
    }
}

void fft_recursive(Complex *data, const size_t n) {
    if (n == 1) {
        return;
    }

    const size_t half = n / 2;
    fft_recursive(data, half);
    fft_recursive(data + half, half);

    transform(data, n);
}

void fft_recursive_stepped(const Complex *input, Complex *output, const size_t n, const size_t step) {
    if (n == 1) {
        output[0] = input[0];
        return;
    }

    const size_t half = n / 2;
    fft_recursive_stepped(input, output, half, step * 2);
    fft_recursive_stepped(input + step, output + half, half, step * 2);

    transform(output, n);
}

void fft_recursive_tasked(Complex *data, const size_t n) {
    if (n == 1) {
        return;
    }

    const size_t half = n / 2;
#pragma omp task
    fft_recursive_tasked(data, half);
#pragma omp task
    fft_recursive_tasked(data + half, half);
#pragma omp taskwait

    transform(data, n);
}

void fft_iterative(Complex *data, const std::size_t n) {
    const std::size_t log2_n = std::countr_zero(n);

    for (std::size_t s = 1; s <= log2_n; ++s) {
        const std::size_t m = 1 << s;
        const std::size_t half_m = m / 2;
        const Complex wm = std::polar(1.0, -2.0 * std::numbers::pi_v<double> / static_cast<double>(m));

        for (std::size_t k = 0; k < n; k += m) {
            Complex w = {1.0, 0.0};
            for (std::size_t j = 0; j < half_m; ++j) {
                Complex t = w * data[k + j + half_m];
                Complex u = data[k + j];
                data[k + j] = u + t;
                data[k + j + half_m] = u - t;
                w *= wm;
            }
        }
    }
}

void fft_iterative_parallel(Complex *data, const std::size_t n) {
    const std::size_t log2_n = std::countr_zero(n);

    for (std::size_t s = 1; s <= log2_n; ++s) {
        const std::size_t m = 1 << s;
        const std::size_t half_m = m / 2;
        const Complex wm = std::polar(1.0, -2.0 * std::numbers::pi_v<double> / static_cast<double>(m));

#pragma omp parallel for schedule(static)
        for (std::size_t k = 0; k < n; k += m) {
            Complex w = {1.0, 0.0};
            for (std::size_t j = 0; j < half_m; ++j) {
                Complex t = w * data[k + j + half_m];
                Complex u = data[k + j];
                data[k + j] = u + t;
                data[k + j + half_m] = u - t;
                w *= wm;
            }
        }
    }
}

void fft(const Complex *input, Complex *output, const size_t n, const Algorithm algorithm) {
    if ((n & n - 1) != 0) {
        throw std::invalid_argument("Size of input array must be a power of 2");
    }

    switch (algorithm) {
        case RECURSIVE:
            bit_shuffle(input, output, n);
            fft_recursive(output, n);
            break;
        case RECURSIVE_STEPPED:
            fft_recursive_stepped(input, output, n, 1);
            break;
        case RECURSIVE_TASKED:
            bit_shuffle(input, output, n);
            fft_recursive_tasked(output, n);
            break;
        case ITERATIVE:
            bit_shuffle(input, output, n);
            fft_iterative(output, n);
            break;
        case ITERATIVE_PARALLEL:
            bit_shuffle(input, output, n);
            fft_iterative_parallel(output, n);
            break;
    }
}
