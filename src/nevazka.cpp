#include "nevazka.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

bool residualInfinityNorm(
    std::size_t n,
    const std::vector<double>& original,
    const std::vector<double>& inverse,
    double& norm
) {
    if (n == 0 || n > std::numeric_limits<std::size_t>::max() / n) {
        return false;
    }

    const std::size_t elementCount = n * n;
    if (original.size() != elementCount || inverse.size() != elementCount) {
        return false;
    }

    norm = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double rowSum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            double product = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                product += original[i * n + k] * inverse[k * n + j];
                if (!std::isfinite(product)) {
                    return false;
                }
            }

            const double residual = product - (i == j ? 1.0 : 0.0);
            if (!std::isfinite(residual)) {
                return false;
            }
            rowSum += std::abs(residual);
            if (!std::isfinite(rowSum)) {
                return false;
            }
        }
        if (rowSum > norm) {
            norm = rowSum;
        }
    }
    return true;
}
