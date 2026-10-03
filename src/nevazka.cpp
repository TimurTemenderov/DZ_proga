#include "nevazka.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

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
    std::vector<double> productRow(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double* originalRow = original.data() + i * n;
        std::fill(productRow.begin(), productRow.end(), 0.0);

        for (std::size_t k = 0; k < n; ++k) {
            const double coefficient = originalRow[k];
            const double* inverseRow = inverse.data() + k * n;
            for (std::size_t j = 0; j < n; ++j) {
                productRow[j] += coefficient * inverseRow[j];
            }
        }

        double rowSum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            if (!std::isfinite(productRow[j])) {
                return false;
            }

            const double residual = productRow[j] - (i == j ? 1.0 : 0.0);
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
