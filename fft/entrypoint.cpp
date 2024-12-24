#include <complex>
#include <bit>
#include <bitset>
#include <iostream>
#include <vector>

void bit_shuffle(const std::complex<double> *in, std::complex<double> *out, const size_t n) {
    // Определяем количество битов для реверса
    const size_t log2_n = std::countr_zero(n);

    // Проверяем, что n является степенью двойки
    if (1u << log2_n != n) {
        throw std::invalid_argument("Размер n должен быть степенью двойки.");
    }

    // Выполняем перестановку в порядке битового реверса
    for (size_t i = 0; i < n; ++i) {
        const size_t reversed_index = std::bitset<sizeof(size_t) * 8>(i).to_ulong() >> sizeof(size_t) * 8 - log2_n;
        out[reversed_index] = in[i];
    }
}

// равзернуть рекурсию в цикл
void fft(const std::complex<double> *input, std::complex<double> *output, const size_t n) {
    if (n == 1) {
        output[0] = input[0];
        return;
    }

    bit_shuffle(input, output, n);

    fft(input, output, n / 2);
    fft(input + n / 2, output + n / 2, n / 2);

    for (std::size_t i = 0; i < n / 2; i++) {
        auto W = std::polar(1.0, -2.0 * i * std::numbers::pi_v<double> / n);
        auto t1 = output[i];
        auto t2 = output[i + n / 2];
        output[i] = t1 + W * t2;
        output[i + n / 2] = t1 - W * t2;
    }
}

// равзернуть рекурсию в цикл
void fft_par(const std::complex<double> *input, std::complex<double> *output, const size_t n) {
    if (n == 1) {
        output[0] = input[0];
        return;
    }

    bit_shuffle(input, output, n);

#pragma omp task
    fft(input, output, n / 2);
#pragma omp task
    fft(input + n / 2, output + n / 2, n / 2);
#pragma omp taskwait

    for (std::size_t i = 0; i < n / 2; i++) {
        auto W = std::polar(1.0, -2.0 * i * std::numbers::pi_v<double> / n);
        auto t1 = output[i];
        auto t2 = output[i + n / 2];
        output[i] = t1 + W * t2;
        output[i + n / 2] = t1 - W * t2;
    }
}

int main(int args, char **argv) {
    // constexpr size_t n = 1u << 8;
    constexpr size_t n = 10;
    std::vector<double> original(n), spectre(n), restored(n);

    for (size_t i = 0; i < n / 2; ++i) {
        original[i] = static_cast<double>(i);
        original[n - i - 1] = original[i];
    }

    for (size_t i = 0; i < n; ++i)
        std::cout << original[i] << " " << std::endl; {
    }
}
