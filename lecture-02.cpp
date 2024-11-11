#include <iostream>
#include <memory>
#include <immintrin.h>
#include <chrono>
#include <algorithm>

// #define PRINT_MATRIX
// #define N 10
// #define M 10
#define N 8096
#define M 8096

double A[N * M];
double B[N * M];
double C[N * M];

void add_matrix(double *A, const double *B, const double *C, const size_t rows, const size_t cols) {
    for (size_t i = 0; i < rows * cols; ++i) {
        A[i] = B[i] + C[i];
    }
}

void add_matrix_avx_unaligned(double *A, const double *B, const double *C, const size_t rows, const size_t cols) {
    for (size_t i = 0; i < rows * cols; i += 4) {
        const __m256d b = _mm256_loadu_pd(&B[i]);
        const __m256d c = _mm256_loadu_pd(&C[i]);
        const __m256d a = _mm256_add_pd(b, c);
        _mm256_storeu_pd(&A[i], a);
    }
}

void print_matrix(const double *A, const size_t rows, const size_t cols) {
    for (size_t i = 0; i < rows * cols; ++i) {
        if (i != 0 && i % cols == 0) {
            std::cout << std::endl;
        }
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;
}

void measure_add_matrix(void (multiplier)(double *, const double *, const double *, size_t, size_t)) {
    std::fill_n(&A[0], N * M, 0);
    std::fill_n(&B[0], N * M, 1);
    std::fill_n(&C[0], N * M, 1);

    const auto t1 = std::chrono::steady_clock::now();
    multiplier(A, B, C, N, M);
    const auto t2 = std::chrono::steady_clock::now();

#ifdef PRINT_MATRIX
    std::cout << "Matrix A:" << std::endl;
    print_matrix(A.get(), N, M);
    std::cout << "Matrix B:" << std::endl;
    print_matrix(B.get(), N, M);
    std::cout << "Matrix C:" << std::endl;
    print_matrix(C.get(), N, M);
#endif
    std::cout << "Time took: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;
}


int main() {
    measure_add_matrix(add_matrix);
    measure_add_matrix(add_matrix_avx_unaligned);

    return 0;
}
