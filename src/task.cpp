#include "task.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

void swapRows(std::vector<double>& matrix, std::size_t n, std::size_t first, std::size_t second) {
    double* firstRow = matrix.data() + first * n;
    double* secondRow = matrix.data() + second * n;
    for (std::size_t j = 0; j < n; ++j) {
        std::swap(firstRow[j], secondRow[j]);
    }
}

void restoreRowOrder(std::size_t n, std::vector<double>& inverse, const std::vector<std::size_t>& permutation) {
    std::vector<unsigned char> visited(n, 0);
    std::vector<double> row(n);

    for (std::size_t start = 0; start < n; ++start) {
        if (visited[start] != 0) {
            continue;
        }

        const double* startRow = inverse.data() + start * n;
        for (std::size_t j = 0; j < n; ++j) {
            row[j] = startRow[j];
        }

        std::size_t source = start;
        do {
            visited[source] = 1;
            const std::size_t destination = permutation[source];
            double* destinationRow = inverse.data() + destination * n;
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(row[j], destinationRow[j]);
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
            const double* row = matrix.data() + i * n;
            for (std::size_t j = k; j < n; ++j) {
                const double magnitude = std::abs(row[j]);
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
                double* row = matrix.data() + i * n;
                std::swap(row[k], row[pivotColumn]);
            }
            std::swap(columnPermutation[k], columnPermutation[pivotColumn]);
        }

        double* pivotMatrixRow = matrix.data() + k * n;
        double* pivotInverseRow = inverse.data() + k * n;
        const double pivot = pivotMatrixRow[k];
        if (!std::isfinite(pivot) || std::abs(pivot) <= tolerance) {
            return InversionStatus::singular;
        }

        for (std::size_t i = k + 1; i < n; ++i) {
            double* row = matrix.data() + i * n;
            double* inverseRow = inverse.data() + i * n;
            const double multiplier = row[k] / pivot;
            if (!std::isfinite(multiplier)) {
                return InversionStatus::numerical_failure;
            }

            for (std::size_t j = k + 1; j < n; ++j) {
                row[j] -= multiplier * pivotMatrixRow[j];
            }
            row[k] = 0.0;

            for (std::size_t j = 0; j < n; ++j) {
                inverseRow[j] -= multiplier * pivotInverseRow[j];
            }
        }
    }

    for (std::size_t ii = n; ii-- > 0;) {
        double* inverseRow = inverse.data() + ii * n;
        const double* row = matrix.data() + ii * n;
        for (std::size_t r = ii + 1; r < n; ++r) {
            const double coefficient = row[r];
            const double* solvedRow = inverse.data() + r * n;
            for (std::size_t j = 0; j < n; ++j) {
                inverseRow[j] -= coefficient * solvedRow[j];
            }
        }

        const double invDiagonal = 1.0 / row[ii];
        for (std::size_t j = 0; j < n; ++j) {
            inverseRow[j] *= invDiagonal;
        }
    }

    restoreRowOrder(n, inverse, columnPermutation);
    for (const double value : inverse) {
        if (!std::isfinite(value)) {
            return InversionStatus::numerical_failure;
        }
    }
    return InversionStatus::success;
}
