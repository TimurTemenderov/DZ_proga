#pragma once

#include <cstddef>
#include <vector>

enum class InversionStatus {
    success,
    singular,
    numerical_failure,
    invalid_input,
    thread_failure
};

// PAQ = LU с полным выбором на каждом шаге, затем блочное решение LY=I, UZ=Y.
// matrix заменяется факторами L/U; inverse получает QZP в исходном порядке.
// matrix и inverse должны быть разными буферами. threadCount включает вызывающий поток.
InversionStatus invertMatrixFullPivot(
    std::size_t n,
    std::vector<double>& matrix,
    std::vector<double>& inverse,
    std::vector<std::size_t>& columnPermutation,
    int threadCount
);
