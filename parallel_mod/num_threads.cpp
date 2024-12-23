#include "num_threads.h"
#include <omp.h> //MSVC: /openmp, gcc: -fopenmp

static unsigned thread_num = 1;

EXTERN_C void set_num_threads(unsigned T)
{
    if (!T || T > static_cast<unsigned>(omp_get_num_procs()))
        T = static_cast<unsigned>(omp_get_num_procs());
    thread_num = T;
    omp_set_num_threads(static_cast<int>(T));
}

EXTERN_C unsigned get_num_threads() {
    return thread_num;
}
