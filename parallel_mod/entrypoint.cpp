#include "vector_mod.h"
#include "test.h"
#include "performance.h"
#include <iostream>
#include <iomanip>

int main(int argc, char **argv) {
    std::cout << "Correctness tests" << std::endl;
    for (std::size_t iTest = 0; iTest < test_data_count; ++iTest) {
        std::cout << "Test " << iTest;
        if (test_data[iTest].result != vector_mod(test_data[iTest].dividend, test_data[iTest].dividend_size,
                                                  test_data[iTest].divisor)) {
            std::cout << " - FAIL" << std::endl;
            return -1;
        }
        std::cout << " - OK" << std::endl;
    }

    std::cout << "Performance tests. ";
    const auto measurements = run_experiments();
    std::cout << " Done\n";
    std::cout << std::setfill(' ') << std::setw(2) << "T:" << " |" << std::setw(3 + 2 * sizeof(IntegerWord)) << "Value:"
            << " | "
            << std::setw(14) << "Duration, ms:" << " | Acceleration:\n";
    for (std::size_t T = 1; T <= measurements.size(); ++T) {
        std::cout << std::setw(2) << T << " | 0x" << std::setw(2 * sizeof(IntegerWord)) << std::setfill('0') << std::hex
                << measurements[T - 1].result;
        std::cout << " | " << std::setfill(' ') << std::setw(14) << std::dec << measurements[T - 1].time.count();
        std::cout << " | " << (static_cast<double>(measurements[0].time.count()) / measurements[T - 1].time.count()) <<
                "\n";
    }

    return 0;
}
