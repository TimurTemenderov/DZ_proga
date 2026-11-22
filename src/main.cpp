#include "task.hpp"
#include "input.hpp"
#include "output.hpp"
#include "nevazka.hpp"

#include <charconv>
#include <chrono>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

// Принимаем целое число только при полном разборе аргумента.
bool parseInteger(const char* text, int& value) {
    const char* end = text;
    while (*end != '\0') {
        ++end;
    }
    const std::from_chars_result result = std::from_chars(text, end, value);
    return result.ec == std::errc{} && result.ptr == end;
}

// n — порядок матрицы, m — ограничение печати, p — число потоков, k — источник.
void printUsage(const char* executable) {
    std::cerr << "Использование: " << executable << " n m p k [filename]\n";
    std::cerr << "p - число потоков; для k=0 требуется файл с матрицей; для k=1..4 используется формула.\n";
}

// Единый формат ошибки и ненулевой код завершения для командной строки.
int fail(const std::string& message) {
    std::cerr << "Ошибка: " << message << '\n';
    return 1;
}

}

// Управляет этапами программы. Чтение, генерация и печать идут в главном потоке.
int main(int argc, char* argv[]) {
    if (argc < 5 || argc > 6) {
        printUsage(argv[0]);
        return 1;
    }

    int nArgument = 0;
    int m = 0;
    int p = 0;
    int k = 0;
    if (!parseInteger(argv[1], nArgument) || !parseInteger(argv[2], m)
        || !parseInteger(argv[3], p) || !parseInteger(argv[4], k)) {
        printUsage(argv[0]);
        return fail("аргументы n, m, p и k должны быть целыми числами");
    }
    if (nArgument <= 0 || m < 0 || p <= 0 || k < 0 || k > 4) {
        return fail("требуется n > 0, m >= 0, p > 0 и 0 <= k <= 4");
    }
    if ((k == 0 && argc != 6) || (k != 0 && argc != 5)) {
        printUsage(argv[0]);
        return fail("имя файла требуется только при k=0");
    }

    const std::size_t n = static_cast<std::size_t>(nArgument);
    // Сначала проверяем арифметику размера, затем предел самого контейнера.
    if (n > std::numeric_limits<std::size_t>::max() / n) {
        return fail("размер n*n превышает допустимый размер контейнера");
    }
    const std::size_t elementCount = n * n;
    const std::vector<double> emptyVector;
    if (elementCount > emptyVector.max_size()) {
        return fail("размер матрицы превышает максимально допустимый размер контейнера");
    }

    const std::string filename = k == 0 ? argv[5] : std::string{};
    try {
        std::vector<double> matrix(elementCount);
        std::vector<double> inverse(elementCount);
        std::vector<std::size_t> columnPermutation(n);
        std::string error;

        if (!initializeMatrix(nArgument, k, filename, matrix, error)) {
            return fail(error);
        }

        std::cout << "Исходная матрица A:\n";
        printMatrix(matrix, n, n, m);
        std::cout.flush();

        // Замер включает только вызов обращения, в том числе запуск его потоков.
        const auto start = std::chrono::steady_clock::now();
        const InversionStatus status = invertMatrixFullPivot(n, matrix, inverse, columnPermutation, p);
        const auto finish = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = finish - start;

        if (status == InversionStatus::singular) {
            return fail("матрица вырождена или численно необратима");
        }
        if (status == InversionStatus::numerical_failure) {
            return fail("численная ошибка при обращении матрицы");
        }
        if (status == InversionStatus::thread_failure) {
            return fail("не удалось создать потоки");
        }
        if (status != InversionStatus::success) {
            return fail("некорректные размеры матриц для обращения");
        }

        std::cout << "Обратная матрица A^-1:\n";
        printMatrix(inverse, n, n, m);
        std::cout.flush();

        // При обращении буфер matrix заняли факторы L/U. Для невязки нужна
        // исходная A, поэтому восстанавливаем её из того же источника.
        if (!initializeMatrix(nArgument, k, filename, matrix, error)) {
            return fail(error);
        }

        // Невязка тоже вычисляется по потокам, уже за пределами замера обращения.
        double residual = 0.0;
        if (!residualInfinityNorm(n, matrix, inverse, p, residual)) {
            return fail("не удалось вычислить норму невязки");
        }

        std::cout << "Норма невязки ||A*A^-1-I||_inf: " << std::scientific << residual << '\n';
        std::cout << "Время обращения: " << std::scientific << elapsed.count() << " с\n";
    } catch (const std::bad_alloc&) {
        return fail("недостаточно памяти для матриц");
    } catch (const std::length_error&) {
        return fail("размер матрицы превышает возможности контейнера");
    } catch (const std::exception& exception) {
        return fail(std::string("ошибка выполнения: ") + exception.what());
    }

    return 0;
}
