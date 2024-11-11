#include <iostream>
#include <memory>
#include <chrono>
#include <algorithm>
#include <cassert>
#include <immintrin.h>

// #define PRINT_MATRIX
#define N1 2560
#define M1 256
#define N2 256
#define M2 256
#define AVX_SIZE 4

double A[N1 * M2];
double B[M1 * N1];
double C[M2 * N2];

void mul_matrix(double *A, const size_t rA, const size_t cA,
                const double *B, const size_t rB, const size_t cB,
                const double *C, const size_t rC, const size_t cC) {
    assert(cB == rC && cA == cC && rA == rB);

    for (size_t i = 0; i < cA; ++i) {
        for (size_t j = 0; j < rA; ++j) {
            A[i * rA + j] = 0;
            for (size_t k = 0; k < cB; ++k) {
                A[i * rA + j] += B[j + rB * k] * C[i * rC + k];
            }
        }
    }
}

void mul_matrix_avx2(double *A, const size_t rA, const size_t cA,
                     const double *B, const size_t rB, const size_t cB,
                     const double *C, const size_t rC, const size_t cC) {
    assert(cB == rC && cA == cC && rA == rB);

    for (size_t i = 0; i < cA; ++i) {
        for (size_t j = 0; j < rA; j += AVX_SIZE) {
            __m256d sum = _mm256_setzero_pd();
            for (size_t k = 0; k < cB; ++k) {
                const __m256d b = _mm256_loadu_pd(&B[j + rB * k]);
                const __m256d c = _mm256_set1_pd(C[i * rC + k]);
                sum = _mm256_fmadd_pd(b, c, sum);
            }
            _mm256_storeu_pd(&A[i * rA + j], sum);
        }
    }
}

void generate_random_matrix(double *M, const size_t rows, const size_t cols) {
    for (int i = 0; i < cols * rows; ++i) {
        M[i] = std::rand() % 9 + 1;
    }
}

void generate_permutation_matrix(double *M, const size_t rows, const size_t cols) {
    assert(rows == cols);

    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < rows; ++j) {
            M[i * cols + j] = 0.0;
        }
        M[i * cols + (cols - 1 - i)] = 1.0;
    }
}

void print_matrix(const double *A, const size_t rows, const size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << A[i + j * rows] << " ";
        }
        std::cout << std::endl;
    }
}

void measure_mul_matrix(void (multiplier)(double *, size_t, size_t,
                                          const double *, size_t, size_t,
                                          const double *, size_t, size_t)) {
    std::fill_n(&A[0], N1 * M2, 0);
    generate_random_matrix(B, N1, M1);
    generate_permutation_matrix(C, N2, M2);

    const auto t1 = std::chrono::steady_clock::now();
    multiplier(A, N1, M2, B, N1, M1, C, N2, M2);
    const auto t2 = std::chrono::steady_clock::now();

#ifdef PRINT_MATRIX
    std::cout << "Matrix A:" << std::endl;
    print_matrix(A, N1, M2);
    std::cout << "Matrix B:" << std::endl;
    print_matrix(B, N1, M1);
    std::cout << "Matrix C:" << std::endl;
    print_matrix(C, N2, M2);
#endif
    std::cout << "Time took: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;
}


int main() {
    measure_mul_matrix(mul_matrix);
    measure_mul_matrix(mul_matrix_avx2);

    return 0;
}
