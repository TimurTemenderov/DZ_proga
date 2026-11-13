#include "task.hpp"

#include <pthread.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

// Обращение матрицы методом Гаусса с полным выбором главного элемента: P A Q = L U,
// где P и Q - перестановки строк и столбцов. Работа делится между pthread-потоками.
// Все потоки проходят одни и те же шаги исключения k = 0..n-1 и синхронизируются барьером.

namespace {

// Сколько шагов исключения накапливаются, прежде чем они применяются к правой части пакетом.
constexpr std::size_t stepBlockSize = 64;
// Ширина полосы столбцов правой части, которая обрабатывается за один проход (для кэша).
constexpr std::size_t columnTileSize = 128;
// Сколько строк обратного хода обрабатываются одним блоком (для кэша).
constexpr std::size_t backBlockSize = 64;

// Кандидат на ведущий элемент: модуль, позиция в матрице и признак нечисловых значений.
struct PivotCandidate {
    double magnitude = 0.0;
    std::size_t row = 0;
    std::size_t column = 0;
    bool failed = false;  // встретились NaN или бесконечность
};

// Данные, общие для всех потоков одного вызова обращения.
struct SharedData {
    std::size_t n = 0;
    std::size_t threadCount = 0;
    double* matrix = nullptr;                  // A; после исключения здесь L (под диагональю) и U
    double* inverse = nullptr;                 // правая часть: сначала единичная, в конце обратная матрица
    std::size_t* columnPermutation = nullptr;  // перестановка столбцов, накопленная при выборе
    double tolerance = 0.0;                    // ведущий элемент не больше этого порога считается нулём

    pthread_barrier_t barrier;                 // все потоки ждут друг друга между фазами шага
    pthread_mutex_t startMutex;                // главный поток держит его, пока создаёт все потоки
    bool startFailed = false;                  // true, если pthread_create не смог создать поток

    std::vector<PivotCandidate> candidates;    // лучший кандидат каждого потока на текущем шаге
    InversionStatus status = InversionStatus::success;  // результат выбора ведущего элемента
};

// Аргументы функции потока: общие данные и номер потока.
struct ThreadArgs {
    SharedData* data = nullptr;
    std::size_t id = 0;
};

// Делит полуинтервал [first, last) на threadCount почти равных частей и возвращает часть номер id как [begin, end).
void threadRange(
    std::size_t first,
    std::size_t last,
    std::size_t threadCount,
    std::size_t id,
    std::size_t& begin,
    std::size_t& end
) {
    const std::size_t length = last - first;
    begin = first + length * id / threadCount;
    end = first + length * (id + 1) / threadCount;
}

// Меняет местами две строки матрицы размера n x n.
void swapRows(double* matrix, std::size_t n, std::size_t first, std::size_t second) {
    double* firstRow = matrix + first * n;
    double* secondRow = matrix + second * n;
    for (std::size_t j = 0; j < n; ++j) {
        std::swap(firstRow[j], secondRow[j]);
    }
}

// Возвращает строки обратной матрицы в исходный порядок по перестановке столбцов, накопленной при выборе.
// Перестановка раскладывается на циклы: строки каждого цикла переставляются по кругу,
// а буфер row хранит строку, которая переносится на следующее место цикла.
// visited отмечает строки, уже обработанные в каком-то цикле.
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

// Максимум модуля элементов строки row, начиная с индекса first.
// Четыре независимых накопителя позволяют процессору и компилятору обрабатывать строку быстрее.
double rowMaxMagnitude(const double* row, std::size_t first, std::size_t n) {
    double max0 = 0.0;
    double max1 = 0.0;
    double max2 = 0.0;
    double max3 = 0.0;
    std::size_t j = first;
    for (; j + 4 <= n; j += 4) {
        max0 = std::max(max0, std::abs(row[j]));
        max1 = std::max(max1, std::abs(row[j + 1]));
        max2 = std::max(max2, std::abs(row[j + 2]));
        max3 = std::max(max3, std::abs(row[j + 3]));
    }
    for (; j < n; ++j) {
        max0 = std::max(max0, std::abs(row[j]));
    }
    return std::max(std::max(max0, max1), std::max(max2, max3));
}

// Обновляет кандидата по строке i (столбцы от first до n-1).
// Сначала берётся максимум строки целиком; позиция элемента ищется только если строка даёт элемент больше текущего.
// Нечисловое значение помечает кандидата как failed и возвращает false.
bool scanRow(const double* row, std::size_t i, std::size_t first, std::size_t n, PivotCandidate& candidate) {
    const double rowMax = rowMaxMagnitude(row, first, n);
    if (!std::isfinite(rowMax)) {
        candidate.failed = true;
        return false;
    }
    if (!(rowMax > candidate.magnitude)) {
        return true;
    }

    for (std::size_t j = first; j < n; ++j) {
        const double magnitude = std::abs(row[j]);
        if (magnitude > candidate.magnitude) {
            candidate.magnitude = magnitude;
            candidate.row = i;
            candidate.column = j;
        }
    }
    return true;
}

// Лучший кандидат среди строк [begin, end) по всем столбцам. Вызывается один раз, до первого шага.
PivotCandidate findLocalPivot(const SharedData& data, std::size_t begin, std::size_t end) {
    const std::size_t n = data.n;
    PivotCandidate candidate;
    for (std::size_t i = begin; i < end; ++i) {
        if (!scanRow(data.matrix + i * n, i, 0, n, candidate)) {
            break;
        }
    }
    return candidate;
}

// Выполняется только потоком 0. Выбирает максимальный по модулю элемент среди кандидатов всех потоков
// (при равенстве побеждает кандидат с меньшим номером потока). Если он не больше порога, матрица вырождена.
// Иначе элемент ставится на диагональ шага k: меняются строки (в матрице и в правой части)
// и столбцы (в матрице), а перестановка столбцов запоминается в columnPermutation.
void selectPivot(SharedData& data, std::size_t k) {
    const std::size_t n = data.n;
    PivotCandidate best;
    for (const PivotCandidate& candidate : data.candidates) {
        if (candidate.failed) {
            data.status = InversionStatus::numerical_failure;
            return;
        }
        if (candidate.magnitude > best.magnitude) {
            best = candidate;
        }
    }

    if (best.magnitude <= data.tolerance) {
        data.status = InversionStatus::singular;
        return;
    }

    if (best.row != k) {
        swapRows(data.matrix, n, best.row, k);
        swapRows(data.inverse, n, best.row, k);
    }

    if (best.column != k) {
        for (std::size_t i = 0; i < n; ++i) {
            double* row = data.matrix + i * n;
            std::swap(row[k], row[best.column]);
        }
        std::swap(data.columnPermutation[k], data.columnPermutation[best.column]);
    }
}

// Шаг исключения для строк [begin, end), все они лежат ниже строки k.
// Для каждой строки множитель row[k] / pivot вычитается из элементов правее k,
// а сам множитель записывается в row[k] (это элемент L). Правая часть здесь не меняется,
// её обновляет applyDelayedSteps. Заодно ищется кандидат на ведущий элемент следующего шага.
PivotCandidate eliminateRows(const SharedData& data, std::size_t k, std::size_t begin, std::size_t end) {
    const std::size_t n = data.n;
    const double* pivotMatrixRow = data.matrix + k * n;
    const double pivot = pivotMatrixRow[k];
    PivotCandidate candidate;

    for (std::size_t i = begin; i < end; ++i) {
        double* row = data.matrix + i * n;
        const double multiplier = row[k] / pivot;
        if (!std::isfinite(multiplier)) {
            candidate.failed = true;
            return candidate;
        }

        for (std::size_t j = k + 1; j < n; ++j) {
            row[j] -= multiplier * pivotMatrixRow[j];
        }
        row[k] = multiplier;

        if (!scanRow(row, i, k + 1, n, candidate)) {
            break;
        }
    }
    return candidate;
}

// Применяет к столбцам правой части [begin, end) шаги исключения k из [first, last).
// Множители берутся из уже сохранённых элементов L. Сначала учитываются строки внутри блока (в порядке шагов),
// затем строки ниже блока: для каждого шага блока выполняется один проход по этим строкам.
// Столбцы обрабатываются полосами шириной columnTileSize.
void applyDelayedSteps(
    const SharedData& data,
    std::size_t first,
    std::size_t last,
    std::size_t begin,
    std::size_t end
) {
    const std::size_t n = data.n;
    for (std::size_t tileBegin = begin; tileBegin < end; tileBegin += columnTileSize) {
        const std::size_t tileEnd = std::min(tileBegin + columnTileSize, end);

        for (std::size_t k = first; k < last; ++k) {
            const double* pivotInverseRow = data.inverse + k * n;
            for (std::size_t i = k + 1; i < last; ++i) {
                const double multiplier = data.matrix[i * n + k];
                double* inverseRow = data.inverse + i * n;
                for (std::size_t j = tileBegin; j < tileEnd; ++j) {
                    inverseRow[j] -= multiplier * pivotInverseRow[j];
                }
            }
        }

        for (std::size_t i = last; i < n; ++i) {
            const double* multipliers = data.matrix + i * n;
            double* inverseRow = data.inverse + i * n;
            for (std::size_t k = first; k < last; ++k) {
                const double multiplier = multipliers[k];
                const double* pivotInverseRow = data.inverse + k * n;
                for (std::size_t j = tileBegin; j < tileEnd; ++j) {
                    inverseRow[j] -= multiplier * pivotInverseRow[j];
                }
            }
        }
    }
}

// Обратный ход для столбцов [begin, end) правой части: решает систему U X = Y снизу вверх.
// Строки делятся на блоки по backBlockSize строк и обрабатываются от нижнего блока к верхнему.
// Внутри блока строка получает вклад решённых строк того же блока, затем умножается на обратное
// значение диагонального элемента. Затем вклад всего блока один раз вычитается из строк выше блока.
// Такие строки обрабатываются по четыре сразу: загруженная строка решения используется в четырёх строках.
// Столбцы обрабатываются полосами шириной columnTileSize.
void backSubstitution(const SharedData& data, std::size_t begin, std::size_t end) {
    const std::size_t n = data.n;
    for (std::size_t tileBegin = begin; tileBegin < end; tileBegin += columnTileSize) {
        const std::size_t tileEnd = std::min(tileBegin + columnTileSize, end);
        std::size_t blockEnd = n;
        while (blockEnd > 0) {
            const std::size_t blockBegin = blockEnd > backBlockSize ? blockEnd - backBlockSize : 0;

            // Строки блока снизу вверх: учитываются только решённые строки того же блока.
            for (std::size_t ii = blockEnd; ii-- > blockBegin;) {
                double* inverseRow = data.inverse + ii * n;
                const double* matrixRow = data.matrix + ii * n;
                for (std::size_t r = ii + 1; r < blockEnd; ++r) {
                    const double coefficient = matrixRow[r];
                    const double* solvedRow = data.inverse + r * n;
                    for (std::size_t j = tileBegin; j < tileEnd; ++j) {
                        inverseRow[j] -= coefficient * solvedRow[j];
                    }
                }
                const double invDiagonal = 1.0 / matrixRow[ii];
                for (std::size_t j = tileBegin; j < tileEnd; ++j) {
                    inverseRow[j] *= invDiagonal;
                }
            }

            // Строки выше блока: по четыре строки за проход, остаток обрабатывается по одной.
            std::size_t i = 0;
            for (; i + 4 <= blockBegin; i += 4) {
                const double* matrixRow0 = data.matrix + i * n;
                const double* matrixRow1 = matrixRow0 + n;
                const double* matrixRow2 = matrixRow1 + n;
                const double* matrixRow3 = matrixRow2 + n;
                double* inverseRow0 = data.inverse + i * n;
                double* inverseRow1 = inverseRow0 + n;
                double* inverseRow2 = inverseRow1 + n;
                double* inverseRow3 = inverseRow2 + n;
                for (std::size_t r = blockBegin; r < blockEnd; ++r) {
                    const double coefficient0 = matrixRow0[r];
                    const double coefficient1 = matrixRow1[r];
                    const double coefficient2 = matrixRow2[r];
                    const double coefficient3 = matrixRow3[r];
                    const double* solvedRow = data.inverse + r * n;
                    for (std::size_t j = tileBegin; j < tileEnd; ++j) {
                        const double solvedValue = solvedRow[j];
                        inverseRow0[j] -= coefficient0 * solvedValue;
                        inverseRow1[j] -= coefficient1 * solvedValue;
                        inverseRow2[j] -= coefficient2 * solvedValue;
                        inverseRow3[j] -= coefficient3 * solvedValue;
                    }
                }
            }
            for (; i < blockBegin; ++i) {
                const double* matrixRow = data.matrix + i * n;
                double* inverseRow = data.inverse + i * n;
                for (std::size_t r = blockBegin; r < blockEnd; ++r) {
                    const double coefficient = matrixRow[r];
                    const double* solvedRow = data.inverse + r * n;
                    for (std::size_t j = tileBegin; j < tileEnd; ++j) {
                        inverseRow[j] -= coefficient * solvedRow[j];
                    }
                }
            }

            blockEnd = blockBegin;
        }
    }
}

// Тело потока. Сначала поток ждёт, пока главный поток создаст все потоки, и выходит, если создание не удалось.
// Далее для каждого шага k все потоки выполняют одни и те же фазы:
//   1) барьер: шаг k-1 закончен у всех, кандидаты готовы;
//   2) если k кратно stepBlockSize, каждый поток применяет накопленный блок шагов к своим столбцам
//      правой части, затем барьер: перестановка строк правой части (фаза 3) затрагивает все столбцы;
//   3) поток 0 выбирает ведущий элемент и переставляет строки и столбцы;
//   4) барьер: результат выбора виден всем; при ошибке все потоки выходят на этом же шаге;
//   5) строки ниже k заново делятся между потоками, каждый исключает свою часть и ищет кандидата.
// После цикла применяется последний (возможно неполный) блок шагов и выполняется обратный ход.
// Столбцы правой части закреплены за потоками: каждый поток работает только со своим диапазоном столбцов.
void* invertThread(void* argument) {
    const ThreadArgs* args = static_cast<const ThreadArgs*>(argument);
    SharedData& data = *args->data;
    const std::size_t id = args->id;
    const std::size_t n = data.n;

    pthread_mutex_lock(&data.startMutex);
    const bool startFailed = data.startFailed;
    pthread_mutex_unlock(&data.startMutex);
    if (startFailed) {
        return nullptr;
    }

    // Начальный кандидат: лучший элемент среди строк этого потока по всем столбцам.
    std::size_t begin = 0;
    std::size_t end = 0;
    threadRange(0, n, data.threadCount, id, begin, end);
    data.candidates[id] = findLocalPivot(data, begin, end);

    // Столбцы правой части, закреплённые за этим потоком, совпадают с его первоначальным диапазоном.
    const std::size_t columnBegin = begin;
    const std::size_t columnEnd = end;

    for (std::size_t k = 0; k < n; ++k) {
        // Фаза 1: шаг k-1 закончен у всех потоков.
        pthread_barrier_wait(&data.barrier);
        if (k > 0 && k % stepBlockSize == 0) {
            // Фаза 2: блок шагов [k - stepBlockSize, k) применяется к своим столбцам правой части.
            applyDelayedSteps(data, k - stepBlockSize, k, columnBegin, columnEnd);
            pthread_barrier_wait(&data.barrier);
        }
        // Фаза 3: выбор ведущего элемента, его делает только поток 0.
        if (id == 0) {
            selectPivot(data, k);
        }
        // Фаза 4: результат выбора виден всем потокам.
        pthread_barrier_wait(&data.barrier);
        if (data.status != InversionStatus::success) {
            return nullptr;
        }

        // Фаза 5: исключение в строках ниже k; эти строки заново делятся между потоками.
        threadRange(k + 1, n, data.threadCount, id, begin, end);
        data.candidates[id] = eliminateRows(data, k, begin, end);
    }

    // Последний блок шагов (от кратного stepBlockSize до n) ещё не применён к правой части.
    applyDelayedSteps(data, (n - 1) / stepBlockSize * stepBlockSize, n, columnBegin, columnEnd);
    backSubstitution(data, columnBegin, columnEnd);
    return nullptr;
}

}

// Обращает матрицу matrix размера n x n; результат пишется в inverse, перестановка столбцов - в columnPermutation.
// На каждый вызов создаётся threadCount потоков, все они завершаются до выхода из функции.
InversionStatus invertMatrixFullPivot(
    std::size_t n,
    std::vector<double>& matrix,
    std::vector<double>& inverse,
    std::vector<std::size_t>& columnPermutation,
    int threadCount
) {
    // Проверки: n > 0, n*n не переполняет size_t, threadCount > 0.
    if (n == 0 || n > std::numeric_limits<std::size_t>::max() / n || threadCount <= 0) {
        return InversionStatus::invalid_input;
    }

    // Размеры векторов должны совпадать с n.
    const std::size_t elementCount = n * n;
    if (matrix.size() != elementCount || inverse.size() != elementCount || columnPermutation.size() != n) {
        return InversionStatus::invalid_input;
    }

    // Все элементы должны быть конечными; заодно находим максимальный модуль для порога малости.
    double scale = 0.0;
    for (const double value : matrix) {
        if (!std::isfinite(value)) {
            return InversionStatus::numerical_failure;
        }
        scale = std::max(scale, std::abs(value));
    }

    // Правая часть сначала единичная; перестановка столбцов тоже тождественная.
    std::fill(inverse.begin(), inverse.end(), 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        inverse[i * n + i] = 1.0;
        columnPermutation[i] = i;
    }

    const std::size_t p = static_cast<std::size_t>(threadCount);
    SharedData data;
    data.n = n;
    data.threadCount = p;
    data.matrix = matrix.data();
    data.inverse = inverse.data();
    data.columnPermutation = columnPermutation.data();
    // Порог малости ведущего элемента: 16 * eps * n * max|a_ij|. Элемент не больше порога считается нулём.
    data.tolerance = 16.0 * std::numeric_limits<double>::epsilon()
        * static_cast<double>(n) * scale;
    data.candidates.resize(p);

    std::vector<ThreadArgs> args(p);
    std::vector<pthread_t> threads(p);

    if (pthread_barrier_init(&data.barrier, nullptr, static_cast<unsigned>(threadCount)) != 0) {
        return InversionStatus::thread_failure;
    }
    if (pthread_mutex_init(&data.startMutex, nullptr) != 0) {
        pthread_barrier_destroy(&data.barrier);
        return InversionStatus::thread_failure;
    }

    // Главный поток держит startMutex, пока создаёт потоки, поэтому они стартуют только после создания всех.
    // Если pthread_create падает, уже созданные потоки видят startFailed и завершаются.
    pthread_mutex_lock(&data.startMutex);
    std::size_t created = 0;
    for (; created < p; ++created) {
        args[created].data = &data;
        args[created].id = created;
        if (pthread_create(&threads[created], nullptr, invertThread, &args[created]) != 0) {
            data.startFailed = true;
            break;
        }
    }
    pthread_mutex_unlock(&data.startMutex);

    // Ждём только реально созданные потоки, затем освобождаем примитивы синхронизации.
    for (std::size_t t = 0; t < created; ++t) {
        pthread_join(threads[t], nullptr);
    }
    pthread_mutex_destroy(&data.startMutex);
    pthread_barrier_destroy(&data.barrier);

    if (data.startFailed) {
        return InversionStatus::thread_failure;
    }
    if (data.status != InversionStatus::success) {
        return data.status;
    }

    // Возвращаем строки обратной матрицы в исходный порядок по перестановке столбцов.
    restoreRowOrder(n, inverse, columnPermutation);
    // Итоговая проверка: все элементы обратной матрицы должны быть конечными.
    for (const double value : inverse) {
        if (!std::isfinite(value)) {
            return InversionStatus::numerical_failure;
        }
    }
    return InversionStatus::success;
}
