#include <iostream>
#include <cassert>
#include <immintrin.h>

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

#if defined(__AVX2__)
void mul_matrix_avx2(double *A, const size_t rA, const size_t cA,
                     const double *B, const size_t rB, const size_t cB,
                     const double *C, const size_t rC, const size_t cC) {
    assert(cB == rC && cA == cC && rA == rB);

    for (size_t i = 0; i < cA; ++i) {
        for (size_t j = 0; j < rA / 4; ++j) {
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
#endif

#if defined(__AVX512F__)
void mul_matrix_avx512(double *A, const size_t rA, const size_t cA,
                       const double *B, const size_t rB, const size_t cB,
                       const double *C, const size_t rC, const size_t cC) {
    assert(cB == rC && cA == cC && rA == rB);

    for (size_t i = 0; i < cA; ++i) {
        for (size_t j = 0; j < rA / 8; ++j) {
            __m512d sum = _mm512_setzero_pd();
            for (size_t k = 0; k < cB; ++k) {
                const __m512d b = _mm512_loadu_pd(&B[j + rB * k]);
                const __m512d c = _mm512_set1_pd(C[i * rC + k]);
                sum = _mm512_fmadd_pd(b, c, sum);
            }
            _mm512_storeu_pd(&A[i * rA + j], sum);
        }
    }
}
#endif
