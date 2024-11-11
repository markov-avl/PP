#include <iostream>
#include <omp.h>

#define N 1000000000

double f(const double x) {
    return x * x;
}


double integrate(const double a, const double b) {
    const double dx = (b - a) / N;
    double sum = 0.0;

    for (int i = 0; i < N; ++i) {
        sum += f(a + i * dx);
    }

    return sum * dx;
}


double integrate_omp(const double a, const double b) {
    const double dx = (b - a) / N;
    double sum = 0.0;

#pragma omp parallel
    {
        const unsigned t = omp_get_thread_num();
        const unsigned T = omp_get_num_threads();
        double local_sum = 0.0;

        for (size_t i = t; i < N; i += T) {
            local_sum += f(a + i * dx);
        }

#pragma omp parallel
        {
            sum += local_sum;
        }
    }

    return sum * dx;
}


double integrate_omp_reduction(const double a, const double b) {
    const double dx = (b - a) / N;
    double sum = 0.0;

#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += f(a + i * dx);
    }

    return sum * dx;
}


void measure(double (*func)(double, double)) {
    const double t1 = omp_get_wtime();
    const double result = func(-1.0, 1.0);
    const double t2 = omp_get_wtime();
    std::cout << "Result: " << result << std::endl;
    std::cout << "Time: " << t2 - t1 << std::endl;
}


int main() {
    measure(integrate);
    measure(integrate_omp);
    measure(integrate_omp_reduction);
}
