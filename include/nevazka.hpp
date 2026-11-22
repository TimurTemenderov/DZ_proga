#pragma once

#include <cstddef>
#include <vector>

// Вычисляет ||A*A^-1-I||_inf, распределяя строки между рабочими потоками.
// При успехе возвращает true и записывает норму в norm.
bool residualInfinityNorm(
    std::size_t n,
    const std::vector<double>& original,
    const std::vector<double>& inverse,
    int threadCount,
    double& norm
);
