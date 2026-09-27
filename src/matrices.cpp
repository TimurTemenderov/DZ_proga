#include "matrices.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

double matrixElement(int k, int n, int i, int j) {
    switch (k) {
    case 1:
        return static_cast<double>(n - std::max(i, j) + 1);
    case 2:
        return static_cast<double>(std::max(i, j));
    case 3:
        return static_cast<double>(std::abs(i - j));
    case 4:
        return 1.0 / (static_cast<double>(i) + static_cast<double>(j) - 1.0);
    default:
        return std::numeric_limits<double>::quiet_NaN();
    }
}

bool fillMatrixByFormula(int n, int k, double* matrix, std::size_t elementCount) {
    if (n <= 0 || k < 1 || k > 4 || matrix == nullptr) {
        return false;
    }

    const std::size_t dimension = static_cast<std::size_t>(n);
    if (dimension > std::numeric_limits<std::size_t>::max() / dimension) {
        return false;
    }
    const std::size_t expectedCount = dimension * dimension;
    if (elementCount != expectedCount) {
        return false;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const std::size_t index = static_cast<std::size_t>(i) * dimension + static_cast<std::size_t>(j);
            matrix[index] = matrixElement(k, n, i + 1, j + 1);
        }
    }
    return true;
}
