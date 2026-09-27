#pragma once

#include <cstddef>

double matrixElement(int k, int n, int i, int j);
bool fillMatrixByFormula(int n, int k, double* matrix, std::size_t elementCount);
