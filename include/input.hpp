#pragma once

#include <string>
#include <vector>

// Читает ровно n*n конечных чисел в заранее выделенный буфер.
// При ошибке возвращает false и записывает её описание в error.
bool readMatrixFromFile(int n, const std::string& filename, std::vector<double>& matrix, std::string& error);

// При k=0 берёт данные из файла, при k=1..4 заполняет матрицу по формуле.
bool initializeMatrix(int n, int k, const std::string& filename, std::vector<double>& matrix, std::string& error);
