#pragma once

#include <cstddef>
#include <vector>

// Печатает левый верхний фрагмент с ограничением maxDimension по обеим осям.
// maxDimension <= 0 отключает печать элементов.
void printMatrix(const std::vector<double>& matrix, std::size_t rows, std::size_t columns, int maxDimension);
