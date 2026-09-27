#pragma once

#include <cstddef>
#include <vector>

enum class InversionStatus {
    success,
    singular,
    numerical_failure,
    invalid_input
};

InversionStatus invertMatrixFullPivot(
    std::size_t n,
    std::vector<double>& matrix,
    std::vector<double>& inverse,
    std::vector<std::size_t>& columnPermutation
);
