#include <iostream>
#include <complex>
#include <vector>
#include <chrono>
#include <iomanip>
#include <stdexcept>

#include "implementations.h"

using namespace std;
using namespace chrono;

void run_experiments(size_t max_power) {
    vector<string> algorithm_names = {
        "Recursive",
        "Recursive Stepped",
        "Recursive Tasked",
        "Iterative",
        "Iterative Parallel"
    };

    cout << left << setw(15) << "Size";
    for (const auto& name : algorithm_names) {
        cout << setw(20) << name;
    }
    cout << endl;
    cout << string(15 + algorithm_names.size() * 20, '-') << endl;

    for (size_t power = 1; power <= max_power; ++power) {
        size_t n = 1 << power;
        vector<Complex> input(n), output(n);

        for (size_t i = 0; i < n; ++i) {
            input[i] = Complex(static_cast<double>(i), 0.0);
        }

        cout << left << setw(15) << n;

        for (int alg = RECURSIVE; alg <= ITERATIVE_PARALLEL; ++alg) {
            auto start_time = high_resolution_clock::now();

            try {
                fft(input.data(), output.data(), n, static_cast<Algorithm>(alg));
            } catch (const exception& e) {
                cerr << "Error in algorithm " << algorithm_names[alg] << ": " << e.what() << endl;
                continue;
            }

            auto end_time = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end_time - start_time).count();

            cout << setw(20) << duration;
        }

        cout << endl;
    }
}

int main() {
    size_t max_power = 22; // Максимальная степень 2

    try {
        run_experiments(max_power);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }

    return 0;
}
