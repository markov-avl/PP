#include <immintrin.h>

void add_matrix(double *A, const double *B, const double *C, const size_t rows, const size_t cols) {
    for (size_t i = 0; i < rows * cols; ++i) {
        A[i] = B[i] + C[i];
    }
}

#if defined(__AVX2__)
void add_matrix_avx2(double *A, const double *B, const double *C, const size_t rows, const size_t cols) {
    for (size_t i = 0; i < rows * cols; i += 4) {
        const __m256d b = _mm256_loadu_pd(&B[i]);
        const __m256d c = _mm256_loadu_pd(&C[i]);
        const __m256d a = _mm256_add_pd(b, c);
        _mm256_storeu_pd(&A[i], a);
    }
}
#endif

#if defined(__AVX512F__)
void add_matrix_avx512(double *A, const double *B, const double *C, const size_t rows, const size_t cols) {
    for (size_t i = 0; i < rows * cols; i += 8) {
        const __m512d b = _mm512_loadu_pd(&B[i]);
        const __m512d c = _mm512_loadu_pd(&C[i]);
        const __m512d a = _mm512_add_pd(b, c);
        _mm512_storeu_pd(&A[i], a);
    }
}
#endif