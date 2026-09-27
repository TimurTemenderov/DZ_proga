#include "output.hpp"

#include <algorithm>
#include <cstdio>
#include <cstddef>
#include <limits>

void printMatrix(const std::vector<double>& matrix, std::size_t rows, std::size_t columns, int maxDimension) {
    if (maxDimension <= 0 || columns == 0) {
        return;
    }
    if (rows > std::numeric_limits<std::size_t>::max() / columns) {
        return;
    }

    const std::size_t limit = static_cast<std::size_t>(maxDimension);
    const std::size_t displayedRows = std::min(rows, limit);
    const std::size_t displayedColumns = std::min(columns, limit);

    for (std::size_t i = 0; i < displayedRows; ++i) {
        for (std::size_t j = 0; j < displayedColumns; ++j) {
            const std::size_t index = i * columns + j;
            if (index >= matrix.size()) {
                return;
            }
            std::printf(" %10.3e", matrix[index]);
        }
        std::printf("\n");
    }
}
