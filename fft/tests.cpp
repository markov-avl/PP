#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <fftw3.h>

#include "implementations.h"


bool check_with_fftw(const std::vector<Complex> &input, const std::string &test_name, Algorithm algorithm) {
    const size_t n = input.size();
    std::vector<Complex> output_custom(n);
    fft(input.data(), output_custom.data(), n, algorithm);

    auto *in = static_cast<fftw_complex *>(fftw_malloc(sizeof(fftw_complex) * n));
    auto *out = static_cast<fftw_complex *>(fftw_malloc(sizeof(fftw_complex) * n));
    fftw_plan p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (size_t i = 0; i < n; ++i) {
        in[i][0] = input[i].real();
        in[i][1] = input[i].imag();
    }

    fftw_execute(p);

    bool correct = true;
    for (size_t i = 0; i < n; ++i) {
        if (std::complex fftw_result(out[i][0], out[i][1]); std::abs(output_custom[i] - fftw_result) > 1e-6) {
            correct = false;
            std::cout << "Test '" << test_name << "' failed at index " << i << ": "
                    << output_custom[i] << " != " << fftw_result << "\n";
        }
    }

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    return correct;
}

void run_all_tests(const Algorithm algorithm) {
    bool passed = true;

    // Тест 1: Единичный импульс
    {
        std::vector<Complex> input = {{1, 0}, {0, 0}, {0, 0}, {0, 0}};
        if (!check_with_fftw(input, "Impulse Test", algorithm)) {
            passed = false;
        } else {
            std::cout << "Impulse Test passed." << std::endl;
        }
    }

    // Тест 2: Константный сигнал
    {
        std::vector<Complex> input = {{1, 0}, {1, 0}, {1, 0}, {1, 0}};
        if (!check_with_fftw(input, "Constant Signal Test", algorithm)) {
            passed = false;
        } else {
            std::cout << "Constant Signal Test passed." << std::endl;
        }
    }

    // Тест 3: Синусоида
    {
        constexpr size_t n = 1048576;
        std::vector<Complex> input(n);
        for (size_t k = 0; k < n; ++k) {
            input[k] = std::polar(1.0, 2 * M_PI * k / n);
        }
        if (!check_with_fftw(input, "Sinusoidal Signal Test", algorithm)) {
            passed = false;
        } else {
            std::cout << "Sinusoidal Signal Test passed." << std::endl;
        }
    }

    // Тест 4: Случайные комплексные числа
    {
        constexpr size_t n = 1048576;
        std::vector<Complex> input(n);
        for (size_t i = 0; i < n; ++i) {
            input[i] = {static_cast<double>(rand()) / RAND_MAX, static_cast<double>(rand()) / RAND_MAX};
        }
        if (!check_with_fftw(input, "Random Complex Numbers Test", algorithm)) {
            passed = false;
        } else {
            std::cout << "Random Complex Numbers Test passed." << std::endl;
        }
    }

    if (passed) {
        std::cout << "All tests passed!" << std::endl;
    } else {
        std::cout << "Some tests failed." << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    run_all_tests(RECURSIVE);
    run_all_tests(RECURSIVE_STEPPED);
    run_all_tests(RECURSIVE_TASKED);
    run_all_tests(ITERATIVE);
    run_all_tests(ITERATIVE_PARALLEL);
    return 0;
}
