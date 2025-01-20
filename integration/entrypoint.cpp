#include <iostream>
#include <iomanip>   // std::setprecision
#include <thread>    // std::thread::hardware_concurrency
#include <omp.h>

// Количество разбиений
static constexpr long long N = 100000000;

// Функция для интегрирования
double f(double x) {
    return x * x;
}

// Однопоточная версия интеграции
double integrate(double a, double b) {
    double dx = (b - a) / N;
    double sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
        sum += f(a + i * dx);
    }
    return sum * dx;
}

// Версия с ручным распараллеливанием OpenMP
double integrate_omp(double a, double b) {
    double dx = (b - a) / N;
    double sum = 0.0;

#pragma omp parallel
    {
        // локальная сумма для каждого потока
        double local_sum = 0.0;
        unsigned t = omp_get_thread_num();
        unsigned T = omp_get_num_threads();

        for (long long i = t; i < N; i += T) {
            local_sum += f(a + i * dx);
        }

        // Собираем результаты
#pragma omp atomic
        sum += local_sum;
    }

    return sum * dx;
}

// Версия с распараллеливанием и reduction
double integrate_omp_reduction(double a, double b) {
    double dx = (b - a) / N;
    double sum = 0.0;

#pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < N; ++i) {
        sum += f(a + i * dx);
    }

    return sum * dx;
}

// Измеряем время работы (возвращает время в секундах)
double measure_time(double (*func)(double, double)) {
    const double start = omp_get_wtime();
    // Запускаем функцию интегрирования
    func(-1.0, 1.0);
    const double end = omp_get_wtime();
    return end - start;
}

int main() {
    // Определяем максимально доступное число аппаратных потоков
    unsigned int maxThreads = std::thread::hardware_concurrency();
    if (maxThreads == 0) {
        // Если система не может определить, принудительно берём хотя бы 1
        maxThreads = 1;
    }

    // Печатаем заголовок таблицы
    std::cout << std::fixed << std::setprecision(4);
    std::cout << " T  |    t (sec)    |  t_OMP (sec)  | t_OMP_RED(sec) |  A(OMP)  | A(OMP_REDUCTION)\n";
    std::cout << "----+---------------+---------------+----------------+----------+-----------------\n";

    // Запускаем в цикле от 1 до maxThreads
    for (unsigned int T = 1; T <= maxThreads; T++) {
        // Настраиваем количество потоков OpenMP
        omp_set_num_threads(T);

        // Измеряем время каждой реализации
        double time_serial  = measure_time(integrate);                // без OMP
        double time_omp     = measure_time(integrate_omp);            // с OMP
        double time_reduct  = measure_time(integrate_omp_reduction);  // с OMP+reduction

        // Считаем ускорения
        double accel_omp    = time_omp    > 0.0 ? time_serial / time_omp    : 0.0;
        double accel_reduct = time_reduct > 0.0 ? time_serial / time_reduct : 0.0;

        // Выводим строку таблицы
        std::cout << std::setw(2) << T << "  | "
                  << std::setw(13) << time_serial << " | "
                  << std::setw(13) << time_omp << " | "
                  << std::setw(16) << time_reduct << " | "
                  << std::setw(8) << accel_omp << " | "
                  << std::setw(15) << accel_reduct << "\n";
    }

    return 0;
}
