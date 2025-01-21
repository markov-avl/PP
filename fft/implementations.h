#ifndef IMPLEMENTATIONS_H
#define IMPLEMENTATIONS_H

#include <complex>

typedef std::complex<double> Complex;

enum Algorithm {
    RECURSIVE,
    RECURSIVE_STEPPED,
    RECURSIVE_TASKED,
    RECURSIVE_PARALLEL,
    ITERATIVE,
    ITERATIVE_PARALLEL
};

void fft(const Complex *, Complex *, size_t n, Algorithm);


#endif
