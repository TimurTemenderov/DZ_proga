#pragma once

#include <cstddef>
#include <vector>

bool residualInfinityNorm(
    std::size_t n,
    const std::vector<double>& original,
    const std::vector<double>& inverse,
    double& norm
);
