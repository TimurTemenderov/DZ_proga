#pragma once

#include <cstddef>

// Элемент формульной матрицы k=1..4, индексы i и j начинаются с единицы.
double matrixElement(int k, int n, int i, int j);

// Заполняет n*n элементов подряд по строкам. false означает неверные параметры.
bool fillMatrixByFormula(int n, int k, double* matrix, std::size_t elementCount);
