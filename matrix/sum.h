#ifndef SUM_H
#define SUM_H

#include <cstddef>

void add_matrix(double *A, const double *B, const double *C, size_t rows, size_t cols);

#if defined(__AVX2__)
void add_matrix_avx2(double *A, const double *B, const double *C, size_t rows, size_t cols);
#endif

#if defined(__AVX512F__)
void add_matrix_avx512(double *A, const double *B, const double *C, size_t rows, size_t cols);
#endif

#endif // SUM_H
