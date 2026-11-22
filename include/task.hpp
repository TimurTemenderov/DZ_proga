#pragma once

#include <cstddef>
#include <vector>

// Позволяет отличить вырожденность от некорректного входа,
// численной ошибки и сбоя запуска рабочих потоков.
enum class InversionStatus {
    success,
    singular,
    numerical_failure,
    invalid_input,
    thread_failure
};

// PAQ = LU с полным выбором на каждом шаге, затем блочное решение LY=I, UZ=Y.
// matrix заменяется факторами L/U; inverse получает QZP в исходном порядке.
// matrix и inverse должны быть разными буферами. threadCount включает вызывающий поток.
// На входе нужны n*n элементов в каждом буфере и n индексов в columnPermutation.
// columnPermutation[i] хранит исходный номер столбца, оказавшегося на позиции i.
InversionStatus invertMatrixFullPivot(
    std::size_t n,
    std::vector<double>& matrix,
    std::vector<double>& inverse,
    std::vector<std::size_t>& columnPermutation,
    int threadCount
);
