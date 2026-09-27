#include "input.hpp"

#include "matrices.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>

bool readMatrixFromFile(int n, const std::string& filename, std::vector<double>& matrix, std::string& error) {
    if (n <= 0) {
        error = "Размер матрицы должен быть положительным";
        return false;
    }

    const std::size_t dimension = static_cast<std::size_t>(n);
    if (dimension > std::numeric_limits<std::size_t>::max() / dimension) {
        error = "Размер матрицы слишком велик";
        return false;
    }
    const std::size_t expectedCount = dimension * dimension;
    if (matrix.size() != expectedCount) {
        error = "Внутренняя ошибка: размер буфера матрицы не совпадает с n*n";
        return false;
    }

    std::ifstream input(filename);
    if (!input) {
        error = "Не удалось открыть файл: " + filename;
        return false;
    }

    for (std::size_t index = 0; index < expectedCount; ++index) {
        std::string token;
        if (!(input >> token)) {
            if (input.eof()) {
                error = "В файле " + filename + " меньше " + std::to_string(expectedCount) + " элементов";
            } else {
                error = "Ошибка чтения файла: " + filename;
            }
            return false;
        }

        char* end = nullptr;
        const double value = std::strtod(token.c_str(), &end);
        if (end == token.c_str() || *end != '\0' || !std::isfinite(value)) {
            error = "В файле " + filename + " встретилось значение неверного формата: " + token;
            return false;
        }
        matrix[index] = value;
    }

    std::string extraToken;
    if (input >> extraToken) {
        error = "В файле " + filename + " больше " + std::to_string(expectedCount) + " элементов";
        return false;
    }
    if (input.bad()) {
        error = "Ошибка чтения файла: " + filename;
        return false;
    }
    return true;
}

bool initializeMatrix(int n, int k, const std::string& filename, std::vector<double>& matrix, std::string& error) {
    if (k == 0) {
        return readMatrixFromFile(n, filename, matrix, error);
    }

    if (!fillMatrixByFormula(n, k, matrix.data(), matrix.size())) {
        error = "Не удалось создать матрицу по формуле";
        return false;
    }
    return true;
}
