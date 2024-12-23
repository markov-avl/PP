#ifndef MUL_H
#define MUL_H

#include <cstddef>

void mul_matrix(double *A, size_t rA, size_t cA,
                const double *B, size_t rB, size_t cB,
                const double *C, size_t rC, size_t cC);

#if defined(__AVX2__)
void mul_matrix_avx2(double *A, size_t rA, size_t cA,
                     const double *B, size_t rB, size_t cB,
                     const double *C, size_t rC, size_t cC);
#endif

#if defined(__AVX512F__)
void mul_matrix_avx512(double *A, size_t rA, size_t cA,
                       const double *B, size_t rB, size_t cB,
                       const double *C, size_t rC, size_t cC);
#endif


#endif // MUL_H
