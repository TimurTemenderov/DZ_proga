#include "task.hpp"
#include "matrices.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef GAUSS_TEST_CREATE_FAILURE
// Линкер перенаправляет вызовы pthread_create в обёртку. Она возвращает ошибку
// на выбранной попытке, а остальные вызовы передаёт настоящей функции.
#include <cerrno>
#include <pthread.h>

namespace {
int failCreateAt = -1;
int createCalls = 0;
}

extern "C" int __real_pthread_create(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);
extern "C" int __wrap_pthread_create(pthread_t* thread, const pthread_attr_t* attributes,
    void* (*entry)(void*), void* argument) {
    if (createCalls++ == failCreateAt) {
        return EAGAIN;
    }
    return __real_pthread_create(thread, attributes, entry, argument);
}
#endif

namespace {

// Исключение останавливает тест и передаёт описание нарушенного условия в main.
void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Простой независимый эталон: на каждом шаге непосредственно просматривает
// всю активную подматрицу, не используя кандидатов предыдущего исключения.
void referenceFactorization(std::size_t n, std::vector<double>& a,
    std::vector<std::size_t>& rows, std::vector<std::size_t>& columns) {
    rows.resize(n);
    columns.resize(n);
    std::iota(rows.begin(), rows.end(), 0);
    std::iota(columns.begin(), columns.end(), 0);
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivotRow = k;
        std::size_t pivotColumn = k;
        double maximum = 0.0;
        for (std::size_t i = k; i < n; ++i) {
            for (std::size_t j = k; j < n; ++j) {
                if (std::abs(a[i * n + j]) > maximum) {
                    maximum = std::abs(a[i * n + j]);
                    pivotRow = i;
                    pivotColumn = j;
                }
            }
        }
        require(maximum > 0.0, "singular reference input");
        // Переносим максимум на диагональ и сохраняем обе перестановки.
        for (std::size_t j = 0; j < n; ++j) {
            std::swap(a[k * n + j], a[pivotRow * n + j]);
        }
        for (std::size_t i = 0; i < n; ++i) {
            std::swap(a[i * n + k], a[i * n + pivotColumn]);
        }
        std::swap(rows[k], rows[pivotRow]);
        std::swap(columns[k], columns[pivotColumn]);
        for (std::size_t i = k + 1; i < n; ++i) {
            a[i * n + k] /= a[k * n + k];
            for (std::size_t j = k + 1; j < n; ++j) {
                a[i * n + j] -= a[i * n + k] * a[k * n + j];
            }
        }
    }
}

// Независимый последовательный расчёт невязки с накоплением в long double.
double residual(std::size_t n, const std::vector<double>& left, const std::vector<double>& right) {
    double norm = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        long double rowSum = 0.0L;
        for (std::size_t j = 0; j < n; ++j) {
            long double value = i == j ? -1.0L : 0.0L;
            for (std::size_t k = 0; k < n; ++k) {
                value += static_cast<long double>(left[i * n + k]) * right[k * n + j];
            }
            rowSum += std::abs(value);
        }
        norm = std::max(norm, static_cast<double>(rowSum));
    }
    return norm;
}

// Воспроизводимая матрица с усиленной диагональю и независимо перемешанными
// строками и столбцами. Это даёт обратимый тест с обеими перестановками.
std::vector<double> randomMatrix(std::size_t n, unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    std::vector<std::size_t> rows(n), columns(n);
    std::iota(rows.begin(), rows.end(), 0);
    std::iota(columns.begin(), columns.end(), 0);
    std::shuffle(rows.begin(), rows.end(), generator);
    std::shuffle(columns.begin(), columns.end(), generator);
    std::vector<double> a(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            a[i * n + j] = value(generator) + (rows[i] == columns[j] ? n + 1.0 : 0.0);
        }
    }
    return a;
}

// Сравниваем результат каждого числа потоков с одним и тем же эталоном.
void checkInverse(std::size_t n, const std::vector<double>& original,
    const std::vector<int>& threadCounts, double tolerance = 1e-10) {
    std::vector<double> reference = original;
    std::vector<std::size_t> rows, columns;
    referenceFactorization(n, reference, rows, columns);
    for (int threads : threadCounts) {
        std::vector<double> factors = original;
        // NaN помогает обнаружить элементы ответа, которые остались незаписанными.
        std::vector<double> inverse(n * n, std::numeric_limits<double>::quiet_NaN());
        std::vector<std::size_t> permutation(n);
        const std::string label = "n=" + std::to_string(n) + " p=" + std::to_string(threads);
        require(invertMatrixFullPivot(n, factors, inverse, permutation, threads)
            == InversionStatus::success, label + " inversion failed");
        require(permutation == columns, label + " full-pivot column order changed");
        for (std::size_t i = 0; i < n * n; ++i) {
            require(std::isfinite(inverse[i]), label + " nonfinite or unwritten inverse");
            require(std::abs(factors[i] - reference[i]) <= 1e-12 * (1.0 + std::abs(reference[i])),
                label + " full-pivot LU differs from serial reference");
        }
        // Проверяются обе стороны: это обнаруживает ошибки в обеих перестановках.
        require(residual(n, original, inverse) < tolerance, label + " A*inverse residual");
        require(residual(n, inverse, original) < tolerance, label + " inverse*A residual");
    }
}

// Проверяем статусы для вырожденности, нечисловых результатов и ошибок входа.
void checkFailures() {
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double maximum = std::numeric_limits<double>::max();
    for (int threads : {1, 2, 4}) {
        for (const auto& matrix : std::vector<std::vector<double>>{
                {0, 0, 0, 0}, {1, 2, 2, 4}}) {
            auto a = matrix;
            std::vector<double> inverse(4);
            std::vector<std::size_t> permutation(2);
            require(invertMatrixFullPivot(2, a, inverse, permutation, threads)
                == InversionStatus::singular, "singular input accepted");
        }
        for (const auto& matrix : std::vector<std::vector<double>>{
                {1, nan, 0, 2}, {1, 0, -infinity, 2}, {maximum, maximum, maximum, -maximum}}) {
            auto a = matrix;
            std::vector<double> inverse(4);
            std::vector<std::size_t> permutation(2);
            require(invertMatrixFullPivot(2, a, inverse, permutation, threads)
                == InversionStatus::numerical_failure, "nonfinite input/result accepted");
        }
        auto a = randomMatrix(65, 42);
        // Повторяющаяся строка даёт вырожденность уже после начала исключения.
        std::copy_n(a.data(), 65, a.data() + 64 * 65);
        std::vector<double> inverse(a.size());
        std::vector<std::size_t> permutation(65);
        require(invertMatrixFullPivot(65, a, inverse, permutation, threads)
            == InversionStatus::singular, "rank-deficient block input accepted");
    }
    std::vector<double> a(4, 1.0), inverse(4);
    std::vector<std::size_t> permutation(2);
    require(invertMatrixFullPivot(0, a, inverse, permutation, 1)
        == InversionStatus::invalid_input, "zero dimension accepted");
    require(invertMatrixFullPivot(std::numeric_limits<std::size_t>::max(), a, inverse, permutation, 1)
        == InversionStatus::invalid_input, "overflow dimension accepted");
    require(invertMatrixFullPivot(2, a, inverse, permutation, 0)
        == InversionStatus::invalid_input, "zero threads accepted");
    require(invertMatrixFullPivot(2, a, inverse, permutation, -1)
        == InversionStatus::invalid_input, "negative threads accepted");
    require(invertMatrixFullPivot(3, a, inverse, permutation, 1)
        == InversionStatus::invalid_input, "incorrect buffer accepted");
    require(invertMatrixFullPivot(2, a, a, permutation, 1)
        == InversionStatus::invalid_input, "aliased buffers accepted");
}

// Проверяем отказ до старта всех рабочих потоков и при частично созданной группе.
void checkThreadCreationFailure() {
#ifdef GAUSS_TEST_CREATE_FAILURE
    for (int failure = 0; failure < 3; ++failure) {
        auto a = randomMatrix(65, 456);
        std::vector<double> inverse(a.size());
        std::vector<std::size_t> permutation(65);
        createCalls = 0;
        failCreateAt = failure;
        const auto status = invertMatrixFullPivot(65, a, inverse, permutation, 4);
        failCreateAt = -1;
        require(status == InversionStatus::thread_failure, "pthread_create failure not handled");
    }
    checkInverse(7, randomMatrix(7, 456), {4, 11});
#endif
}

}

// Набор охватывает малые размеры, границы порций и блоков, формулы и ошибки.
int main() {
    try {
        checkInverse(1, {2}, {1, 2, 4});
        checkInverse(2, {1, 2, 3, 8}, {1, 2, 4});
        checkInverse(3, {0, 2, 2, 2, 0, 2, 2, 2, 0}, {1, 2, 3, 4});
        for (std::size_t n : {7, 17, 63, 64, 65, 127, 128, 129, 257}) {
            checkInverse(n, randomMatrix(n, static_cast<unsigned>(n)), {1, 2, 4});
        }
        checkInverse(517, randomMatrix(517, 123), {3, 4});
        // Равные максимумы в разных порциях должны выбираться в порядке строк.
        constexpr std::size_t tieSize = 273;
        std::vector<double> ties(tieSize * tieSize);
        for (std::size_t i = 0; i < tieSize; ++i) {
            ties[i * tieSize + (37 * i + 11) % tieSize] = 1.0;
        }
        checkInverse(tieSize, ties, {3, 4});
        for (int formula = 1; formula <= 3; ++formula) {
            for (int n : {3, 65, 129}) {
                std::vector<double> a(static_cast<std::size_t>(n) * n);
                require(fillMatrixByFormula(n, formula, a.data(), a.size()), "formula generation");
                checkInverse(n, a, {1, 4}, 1e-7);
            }
        }
        std::vector<double> hilbert(25);
        fillMatrixByFormula(5, 4, hilbert.data(), hilbert.size());
        checkInverse(5, hilbert, {1, 4}, 1e-8);
        checkFailures();
        checkThreadCreationFailure();
        std::cout << "All full-pivot, block-boundary and residual checks passed\n";
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
    return 0;
}
