#include <iostream>
#include <iomanip>
#include <chrono>
#include <functional>
#include <numeric>
#include <vector>

#include "sum.h"
#include "mul.h"

// #define AVX512_ON

constexpr size_t MAX_EXP_SUM = 14;
constexpr size_t MAX_EXP_MUL = 10;
constexpr size_t RUNS = 5;

// Функция для вычисления среднего времени выполнения (без учета минимума и максимума)
double calculate_average(std::vector<double> &times) {
    std::sort(times.begin(), times.end());
    if (times.size() > 2) {
        times.erase(times.begin()); // Удаляем минимум
        times.erase(times.end() - 1); // Удаляем максимум
    }
    const double sum = std::accumulate(times.begin(), times.end(), 0.0);
    return sum / static_cast<double>(times.size());
}

// Общая функция для бенчмарка сложения
std::vector<double> run_benchmark_add(const size_t N, const size_t runs,
                                      const std::function<void
                                          (double *, const double *, const double *, size_t, size_t)> &func) {
    // Создаем массивы
    auto *A = new double[N * N];
    auto *B = new double[N * N];
    auto *C = new double[N * N];

    std::vector<double> times;

    // Выполняем прогоны
    for (size_t i = 0; i < runs; ++i) {
        std::fill_n(B, N * N, 1.0);
        std::fill_n(C, N * N, 1.0);

        auto start = std::chrono::high_resolution_clock::now();
        func(A, B, C, N, N);
        auto end = std::chrono::high_resolution_clock::now();

        times.push_back(std::chrono::duration<double>(end - start).count());
    }

    delete[] A;
    delete[] B;
    delete[] C;

    return times;
}

// Общая функция для бенчмарка умножения
std::vector<double> run_benchmark_mul(const size_t N, const size_t runs,
                                      const std::function<void(double *, size_t, size_t,
                                                               const double *, size_t, size_t,
                                                               const double *, size_t, size_t)> &func) {
    // Создаем массивы
    auto *A = new double[N * N];
    auto *B = new double[N * N];
    auto *C = new double[N * N];

    std::vector<double> times;

    // Выполняем прогоны
    for (size_t i = 0; i < runs; ++i) {
        std::fill_n(B, N * N, 1.0);
        std::fill_n(C, N * N, 1.0);

        auto start = std::chrono::high_resolution_clock::now();
        func(A, N, N, B, N, N, C, N, N);
        auto end = std::chrono::high_resolution_clock::now();

        times.push_back(std::chrono::duration<double>(end - start).count());
    }

    delete[] A;
    delete[] B;
    delete[] C;

    return times;
}

void benchmark() {
    const int nxn_width = static_cast<int>(std::to_string(1 << MAX_EXP_SUM).size()) * 2 + 2;

    std::cout << "=== Addition Benchmark ===\n";
    std::cout << std::left << std::setw(nxn_width) << "NxN"
            << std::setw(15) << "Scalar"
#if defined(__AVX2__)
            << std::setw(15) << "AVX2"
            << std::setw(20) << "AVX2 Acceleration"
#endif
#if defined(__AVX512F__) && defined(AVX512_ON)
              << std::setw(15) << "AVX512"
              << std::setw(20) << "AVX512 Acceleration"
#endif
            << std::endl;

    std::cout << std::string(nxn_width + 15, '-')
#if defined(__AVX2__)
            << std::string(35, '-')
#endif
#if defined(__AVX512F__) && defined(AVX512_ON)
              << std::string(35, '-')
#endif
            << std::endl;

    for (size_t exp = 1; exp <= MAX_EXP_SUM; ++exp) {
        size_t N = 1 << exp;

        // Прогоны для сложения
        auto scalar_times = run_benchmark_add(N, RUNS, add_matrix);

#if defined(__AVX2__)
        auto avx2_times = run_benchmark_add(N, RUNS, add_matrix_avx2);
#endif

#if defined(__AVX512F__) && defined(AVX512_ON)
        auto avx512_times = run_benchmark_add(N, RUNS, add_matrix_avx512);
#endif

        const double scalar_avg = calculate_average(scalar_times);
#if defined(__AVX2__)
        const double avx2_avg = calculate_average(avx2_times);
#endif
#if defined(__AVX512F__) && defined(AVX512_ON)
        const double avx512_avg = calculate_average(avx512_times);
#endif

        std::cout << std::left << std::setw(nxn_width) << (std::to_string(N) + "x" + std::to_string(N))
                << std::setw(15) << scalar_avg
#if defined(__AVX2__)
                << std::setw(15) << avx2_avg
                << std::setw(20) << scalar_avg / avx2_avg
#endif
#if defined(__AVX512F__) && defined(AVX512_ON)
                  << std::setw(15) << avx512_avg
                  << std::setw(20) << scalar_avg / avx512_avg
#endif
                << std::endl;
    }

    std::cout << "\n=== Multiplication Benchmark ===\n";
    std::cout << std::left << std::setw(nxn_width) << "NxN"
            << std::setw(15) << "Scalar"
#if defined(__AVX2__)
            << std::setw(15) << "AVX2"
            << std::setw(20) << "AVX2 Acceleration"
#endif
#if defined(__AVX512F__) && defined(AVX512_ON)
              << std::setw(15) << "AVX512"
              << std::setw(20) << "AVX512 Acceleration"
#endif
            << std::endl;

    std::cout << std::string(nxn_width + 15, '-')
#if defined(__AVX2__)
            << std::string(35, '-')
#endif
#if defined(__AVX512F__) && defined(AVX512_ON)
              << std::string(35, '-')
#endif
            << std::endl;

    for (size_t exp = 1; exp <= MAX_EXP_MUL; ++exp) {
        size_t N = 1 << exp;

        // Прогоны для умножения
        auto scalar_times = run_benchmark_mul(N, RUNS, mul_matrix);

#if defined(__AVX2__)
        auto avx2_times = run_benchmark_mul(N, RUNS, mul_matrix_avx2);
#endif

#if defined(__AVX512F__) && defined(AVX512_ON)
        auto avx512_times = run_benchmark_mul(N, RUNS, mul_matrix_avx512);
#endif

        const double scalar_avg = calculate_average(scalar_times);
#if defined(__AVX2__)
        const double avx2_avg = calculate_average(avx2_times);
#endif
#if defined(__AVX512F__) && defined(AVX512_ON)
        const double avx512_avg = calculate_average(avx512_times);
#endif

        std::cout << std::left << std::setw(nxn_width) << (std::to_string(N) + "x" + std::to_string(N))
                << std::setw(15) << scalar_avg
#if defined(__AVX2__)
                << std::setw(15) << avx2_avg
                << std::setw(20) << scalar_avg / avx2_avg
#endif
#if defined(__AVX512F__) && defined(AVX512_ON)
                  << std::setw(15) << avx512_avg
                  << std::setw(20) << scalar_avg / avx512_avg
#endif
                << std::endl;
    }
}

int main() {
    benchmark();
    return 0;
}
