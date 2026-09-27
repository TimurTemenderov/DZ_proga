#include "task.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

void swapRows(std::vector<double>& matrix, std::size_t n, std::size_t first, std::size_t second) {
    for (std::size_t j = 0; j < n; ++j) {
        std::swap(matrix[first * n + j], matrix[second * n + j]);
    }
}

void restoreRowOrder(std::size_t n, std::vector<double>& inverse, const std::vector<std::size_t>& permutation) {
    std::vector<unsigned char> visited(n, 0);
    std::vector<double> row(n);

    for (std::size_t start = 0; start < n; ++start) {
        if (visited[start] != 0) {
            continue;
        }

        for (std::size_t j = 0; j < n; ++j) {
            row[j] = inverse[start * n + j];
        }

        std::size_t source = start;
        do {
            visited[source] = 1;
            const std::size_t destination = permutation[source];
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(row[j], inverse[destination * n + j]);
            }
            source = destination;
        } while (source != start);
    }
}

}

InversionStatus invertMatrixFullPivot(
    std::size_t n,
    std::vector<double>& matrix,
    std::vector<double>& inverse,
    std::vector<std::size_t>& columnPermutation
) {
    if (n == 0 || n > std::numeric_limits<std::size_t>::max() / n) {
        return InversionStatus::invalid_input;
    }

    const std::size_t elementCount = n * n;
    if (matrix.size() != elementCount || inverse.size() != elementCount || columnPermutation.size() != n) {
        return InversionStatus::invalid_input;
    }

    double scale = 0.0;
    for (const double value : matrix) {
        if (!std::isfinite(value)) {
            return InversionStatus::numerical_failure;
        }
        scale = std::max(scale, std::abs(value));
    }

    std::fill(inverse.begin(), inverse.end(), 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        inverse[i * n + i] = 1.0;
        columnPermutation[i] = i;
    }

    const double tolerance = 16.0 * std::numeric_limits<double>::epsilon()
        * static_cast<double>(n) * scale;

    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivotRow = k;
        std::size_t pivotColumn = k;
        double pivotMagnitude = 0.0;

        for (std::size_t i = k; i < n; ++i) {
            for (std::size_t j = k; j < n; ++j) {
                const double magnitude = std::abs(matrix[i * n + j]);
                if (!std::isfinite(magnitude)) {
                    return InversionStatus::numerical_failure;
                }
                if (magnitude > pivotMagnitude) {
                    pivotMagnitude = magnitude;
                    pivotRow = i;
                    pivotColumn = j;
                }
            }
        }

        if (pivotMagnitude <= tolerance) {
            return InversionStatus::singular;
        }

        if (pivotRow != k) {
            swapRows(matrix, n, pivotRow, k);
            swapRows(inverse, n, pivotRow, k);
        }

        if (pivotColumn != k) {
            for (std::size_t i = 0; i < n; ++i) {
                std::swap(matrix[i * n + k], matrix[i * n + pivotColumn]);
            }
            std::swap(columnPermutation[k], columnPermutation[pivotColumn]);
        }

        const double pivot = matrix[k * n + k];
        if (!std::isfinite(pivot) || std::abs(pivot) <= tolerance) {
            return InversionStatus::singular;
        }

        for (std::size_t i = k + 1; i < n; ++i) {
            const double multiplier = matrix[i * n + k] / pivot;
            if (!std::isfinite(multiplier)) {
                return InversionStatus::numerical_failure;
            }

            for (std::size_t j = k + 1; j < n; ++j) {
                const double value = matrix[i * n + j] - multiplier * matrix[k * n + j];
                if (!std::isfinite(value)) {
                    return InversionStatus::numerical_failure;
                }
                matrix[i * n + j] = value;
            }
            matrix[i * n + k] = 0.0;

            for (std::size_t j = 0; j < n; ++j) {
                const double value = inverse[i * n + j] - multiplier * inverse[k * n + j];
                if (!std::isfinite(value)) {
                    return InversionStatus::numerical_failure;
                }
                inverse[i * n + j] = value;
            }
        }
    }

    for (std::size_t ii = n; ii-- > 0;) {
        for (std::size_t j = 0; j < n; ++j) {
            double value = inverse[ii * n + j];
            for (std::size_t r = ii + 1; r < n; ++r) {
                value -= matrix[ii * n + r] * inverse[r * n + j];
            }
            value /= matrix[ii * n + ii];
            if (!std::isfinite(value)) {
                return InversionStatus::numerical_failure;
            }
            inverse[ii * n + j] = value;
        }
    }

    restoreRowOrder(n, inverse, columnPermutation);
    return InversionStatus::success;
}
