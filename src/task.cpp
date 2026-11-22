#include "task.hpp"

#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <thread>
#include <vector>

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

constexpr std::size_t columnTileSize = 128;
constexpr std::size_t solveBlockSize = 32;
constexpr std::size_t eliminationChunkSize = 32;

// Разные потоки записывают кандидатов в разные кэш-линии.
struct alignas(64) PivotCandidate {
    double magnitude = 0.0;
    std::size_t row = 0;
    std::size_t column = 0;
    bool failed = false;
};

struct SharedData {
    std::size_t n = 0;
    std::size_t threadCount = 0;
    std::size_t columnTileCount = 0;
    double* matrix = nullptr;                  // L под диагональю, U на диагонали и выше
    double* inverse = nullptr;
    std::size_t* rowPermutation = nullptr;     // P[i, rowPermutation[i]] = 1
    std::size_t* columnPermutation = nullptr;  // Q[columnPermutation[i], i] = 1
    std::size_t pivotColumn = 0;
    double tolerance = 0.0;

    alignas(64) std::atomic<std::size_t> remaining{0};
    alignas(64) std::atomic<std::size_t> generation{0};
    alignas(64) std::atomic<std::size_t> nextColumnTile{0};
    alignas(64) std::atomic<std::size_t> nextRow{0};
    pthread_mutex_t startMutex;
    bool startFailed = false;
    std::vector<PivotCandidate> candidates;
    InversionStatus status = InversionStatus::success;
};

struct ThreadArgs {
    SharedData* data = nullptr;
    std::size_t id = 0;
    std::size_t stride = 0;
    std::vector<double> rightHandSides;
    bool failed = false;
};

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

void swapRows(double* matrix, std::size_t n, std::size_t first, std::size_t second) {
    double* firstRow = matrix + first * n;
    double* secondRow = matrix + second * n;
    for (std::size_t j = 0; j < n; ++j) {
        std::swap(firstRow[j], secondRow[j]);
    }
}

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

bool updateCandidate(const double* row, std::size_t i, std::size_t first, std::size_t n,
    double rowMax, PivotCandidate& candidate) {
    if (!std::isfinite(rowMax)) {
        candidate.failed = true;
        return false;
    }
    if (rowMax > candidate.magnitude) {
        // Модуль уже известен. Первого совпадения достаточно, чтобы сохранить
        // тот же порядок при равенстве, что и при последовательном обходе.
        for (std::size_t j = first; j < n; ++j) {
            if (std::abs(row[j]) >= rowMax) {
                candidate.magnitude = rowMax;
                candidate.row = i;
                candidate.column = j;
                break;
            }
        }
    }
    return true;
}

// Проверка входа и начальный полный поиск тоже распределены между потоками.
PivotCandidate findLocalPivot(const SharedData& data, std::size_t begin, std::size_t end) {
    PivotCandidate candidate;
    for (std::size_t i = begin; i < end; ++i) {
        const double* row = data.matrix + i * data.n;
        for (std::size_t j = 0; j < data.n; ++j) {
            if (!std::isfinite(row[j])) {
                candidate.failed = true;
                return candidate;
            }
        }
        if (!updateCandidate(row, i, 0, data.n, rowMaxMagnitude(row, 0, data.n), candidate)) {
            break;
        }
    }
    return candidate;
}

// Полный поиск на каждом шаге. При равенстве сохраняется первый элемент
// в порядке строк. При факторизации достаточно запомнить перестановки P и Q.
void selectPivot(SharedData& data, std::size_t k) {
    const std::size_t n = data.n;
    PivotCandidate best;
    for (const PivotCandidate& candidate : data.candidates) {
        if (candidate.failed) {
            data.status = InversionStatus::numerical_failure;
            return;
        }
        // Порции распределяются динамически, поэтому порядок потоков
        // может отличаться от порядка строк в активной подматрице.
        if (candidate.magnitude > best.magnitude
            || (candidate.magnitude >= best.magnitude && candidate.row < best.row)) {
            best = candidate;
        }
    }
    if (k == 0) {
        data.tolerance = 16.0 * std::numeric_limits<double>::epsilon()
            * static_cast<double>(n) * best.magnitude;
    }
    if (best.magnitude <= data.tolerance) {
        data.status = InversionStatus::singular;
        return;
    }
    if (best.row != k) {
        swapRows(data.matrix, n, best.row, k);
        std::swap(data.rowPermutation[k], data.rowPermutation[best.row]);
    }
    data.pivotColumn = best.column;
    if (best.column != k) {
        // Ведущая строка должна быть готова до освобождения потоков.
        double* row = data.matrix + k * n;
        std::swap(row[k], row[best.column]);
        std::swap(data.columnPermutation[k], data.columnPermutation[best.column]);
    }
}

// Барьер с завершающей операцией: последний прибывший поток выбирает элемент
// до освобождения остальных. Таким образом, вместо двух барьеров нужен один.
// acq_rel у счётчика собирает записи всех кандидатов; release/acquire у поколения
// публикует перестановки и status всем участникам. При ожидании уступаем процессор.
void finishFactorizationStep(SharedData& data, std::size_t k, bool useRowQueue) {
    if (data.threadCount == 1) {
        selectPivot(data, k);
        return;
    }
    const std::size_t generation = data.generation.load(std::memory_order_acquire);
    if (data.remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        selectPivot(data, k);
        if (useRowQueue) {
            // Сброс очереди публикуется вместе с ведущей строкой через generation.
            data.nextRow.store(k + 1, std::memory_order_relaxed);
        }
        data.remaining.store(data.threadCount, std::memory_order_relaxed);
        data.generation.fetch_add(1, std::memory_order_release);
    } else {
        std::size_t spins = 0;
        while (data.generation.load(std::memory_order_acquire) == generation) {
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
            _mm_pause();
#endif
            if (++spins == 1024) {
                std::this_thread::yield();
                spins = 0;
            }
        }
    }
}

// Ведущая строка доступна только для чтения до следующего барьера.
// Каждый поток обновляет только полученные им строки активной области.
void updateRows(const SharedData& data, std::size_t k, std::size_t begin,
    std::size_t end, PivotCandidate& candidate) {
    const std::size_t n = data.n;
    const double* pivotRow = data.matrix + k * n;
    for (std::size_t i = begin; i < end; ++i) {
        double* row = data.matrix + i * n;
        // Переставляем столбцы в своей строке до вычисления множителя L.
        if (data.pivotColumn != k) {
            std::swap(row[k], row[data.pivotColumn]);
        }
        const double multiplier = row[k] / pivotRow[k];
        if (!std::isfinite(multiplier)) {
            candidate.failed = true;
            return;
        }
        // Максимум вычисляется при обновлении строки, без второго полного прохода.
        // Независимые накопители позволяют векторизовать арифметику и редукцию.
        double max0 = 0.0;
        double max1 = 0.0;
        double max2 = 0.0;
        double max3 = 0.0;
        std::size_t j = k + 1;
        for (; j + 4 <= n; j += 4) {
            const double value0 = row[j] - multiplier * pivotRow[j];
            const double value1 = row[j + 1] - multiplier * pivotRow[j + 1];
            const double value2 = row[j + 2] - multiplier * pivotRow[j + 2];
            const double value3 = row[j + 3] - multiplier * pivotRow[j + 3];
            row[j] = value0;
            row[j + 1] = value1;
            row[j + 2] = value2;
            row[j + 3] = value3;
            max0 = std::max(max0, std::abs(value0));
            max1 = std::max(max1, std::abs(value1));
            max2 = std::max(max2, std::abs(value2));
            max3 = std::max(max3, std::abs(value3));
        }
        for (; j < n; ++j) {
            const double value = row[j] - multiplier * pivotRow[j];
            row[j] = value;
            max0 = std::max(max0, std::abs(value));
        }
        row[k] = multiplier;
        const double rowMax = std::max(std::max(max0, max1), std::max(max2, max3));
        if (!updateCandidate(row, i, k + 1, n, rowMax, candidate)) {
            break;
        }
    }
}

PivotCandidate eliminateRows(const SharedData& data, std::size_t k, std::size_t begin, std::size_t end) {
    PivotCandidate candidate;
    updateRows(data, k, begin, end, candidate);
    return candidate;
}

// Порции выдаются в порядке строк. Каждый поток сохраняет первого
// своего кандидата при равных модулях.
PivotCandidate eliminateRowChunks(SharedData& data, std::size_t k) {
    PivotCandidate local;
    for (;;) {
        const std::size_t begin = data.nextRow.fetch_add(eliminationChunkSize, std::memory_order_relaxed);
        if (begin >= data.n) {
            return local;
        }
        const std::size_t end = std::min(begin + eliminationChunkSize, data.n);
        // Следующая порция сравнивается с максимумом всех уже обработанных
        // порций этого потока, поэтому слабые кандидаты не требуют поиска столбца.
        updateRows(data, k, begin, end, local);
        if (local.failed) {
            return local;
        }
    }
}

// Вклад решённого блока [first, last) вычитается из строк [begin, end).
// Четыре строки-приёмника используют одну загрузку каждого элемента блока.
void subtractSolvedBlock(
    const SharedData& data,
    double* rhs,
    std::size_t stride,
    std::size_t width,
    std::size_t begin,
    std::size_t end,
    std::size_t first,
    std::size_t last
) {
    const std::size_t n = data.n;
    std::size_t i = begin;
    for (; i + 4 <= end; i += 4) {
        const double* row0 = data.matrix + i * n;
        const double* row1 = row0 + n;
        const double* row2 = row1 + n;
        const double* row3 = row2 + n;
        double* target0 = rhs + i * stride;
        double* target1 = target0 + stride;
        double* target2 = target1 + stride;
        double* target3 = target2 + stride;
        for (std::size_t k = first; k < last; ++k) {
            const double coefficient0 = row0[k];
            const double coefficient1 = row1[k];
            const double coefficient2 = row2[k];
            const double coefficient3 = row3[k];
            const double* solved = rhs + k * stride;
            for (std::size_t j = 0; j < width; ++j) {
                const double value = solved[j];
                target0[j] -= coefficient0 * value;
                target1[j] -= coefficient1 * value;
                target2[j] -= coefficient2 * value;
                target3[j] -= coefficient3 * value;
            }
        }
    }
    for (; i < end; ++i) {
        const double* row = data.matrix + i * n;
        double* target = rhs + i * stride;
        for (std::size_t k = first; k < last; ++k) {
            const double coefficient = row[k];
            const double* solved = rhs + k * stride;
            for (std::size_t j = 0; j < width; ++j) {
                target[j] -= coefficient * solved[j];
            }
        }
    }
}

// L Y = I. Для полосы, начинающейся со столбца firstColumn, все строки
// выше firstColumn нулевые. В диагональных блоках также пропускаем нулевой хвост.
void forwardSubstitution(const SharedData& data, double* rhs, std::size_t stride,
    std::size_t firstColumn, std::size_t width) {
    const std::size_t n = data.n;
    for (std::size_t first = firstColumn; first < n; first += solveBlockSize) {
        const std::size_t last = std::min(first + solveBlockSize, n);
        for (std::size_t i = first; i < last; ++i) {
            const double* row = data.matrix + i * n;
            double* target = rhs + i * stride;
            for (std::size_t k = first; k < i; ++k) {
                const double coefficient = row[k];
                const double* solved = rhs + k * stride;
                const std::size_t activeWidth = std::min(width, k - firstColumn + 1);
                for (std::size_t j = 0; j < activeWidth; ++j) {
                    target[j] -= coefficient * solved[j];
                }
            }
        }
        const std::size_t activeWidth = std::min(width, last - firstColumn);
        subtractSolvedBlock(data, rhs, stride, activeWidth, last, n, first, last);
    }
}

// U Z = Y. Каждый поток независимо решает системы для своих столбцов Z.
void backSubstitution(const SharedData& data, double* rhs, std::size_t stride, std::size_t width) {
    const std::size_t n = data.n;
    std::size_t last = n;
    while (last > 0) {
        const std::size_t first = last > solveBlockSize ? last - solveBlockSize : 0;
        for (std::size_t i = last; i-- > first;) {
            const double* row = data.matrix + i * n;
            double* target = rhs + i * stride;
            for (std::size_t k = i + 1; k < last; ++k) {
                const double coefficient = row[k];
                const double* solved = rhs + k * stride;
                for (std::size_t j = 0; j < width; ++j) {
                    target[j] -= coefficient * solved[j];
                }
            }
            const double invDiagonal = 1.0 / row[i];
            for (std::size_t j = 0; j < width; ++j) {
                target[j] *= invDiagonal;
            }
        }
        subtractSolvedBlock(data, rhs, stride, width, 0, first, first, last);
        last = first;
    }
}

// A^-1 = Q Z P. Более дорогие начальные полосы выдаются первыми.
// Потоки записывают непересекающиеся столбцы исходной матрицы.
void solveInverseColumns(ThreadArgs& args) {
    SharedData& data = *args.data;
    const std::size_t n = data.n;
    const std::size_t stride = args.stride;
    double* rhs = args.rightHandSides.data();
    for (;;) {
        const std::size_t tile = data.nextColumnTile.fetch_add(1, std::memory_order_relaxed);
        if (tile >= data.columnTileCount) {
            break;
        }
        std::size_t first = 0;
        std::size_t last = 0;
        threadRange(0, n, data.columnTileCount, tile, first, last);
        const std::size_t width = last - first;
        std::fill(args.rightHandSides.begin(), args.rightHandSides.end(), 0.0);
        for (std::size_t j = 0; j < width; ++j) {
            rhs[(first + j) * stride + j] = 1.0;
        }
        forwardSubstitution(data, rhs, stride, first, width);
        backSubstitution(data, rhs, stride, width);
        const std::size_t* outputColumns = data.rowPermutation + first;
        const std::size_t firstOutputColumn = outputColumns[0];
        bool consecutive = true;
        for (std::size_t j = 1; j < width; ++j) {
            if (outputColumns[j] != firstOutputColumn + j) {
                consecutive = false;
                break;
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            const double* source = rhs + i * stride;
            for (std::size_t j = 0; j < width; ++j) {
                if (!std::isfinite(source[j])) {
                    args.failed = true;
                    return;
                }
            }
            double* destination = data.inverse + data.columnPermutation[i] * n;
            if (consecutive) {
                std::copy_n(source, width, destination + firstOutputColumn);
            } else {
                for (std::size_t j = 0; j < width; ++j) {
                    destination[outputColumns[j]] = source[j];
                }
            }
        }
    }
}

void* invertThread(void* argument) {
    ThreadArgs& args = *static_cast<ThreadArgs*>(argument);
    SharedData& data = *args.data;
    if (data.threadCount > 1) {
        pthread_mutex_lock(&data.startMutex);
        const bool startFailed = data.startFailed;
        pthread_mutex_unlock(&data.startMutex);
        if (startFailed) {
            return nullptr;
        }
    }
    std::size_t begin = 0;
    std::size_t end = 0;
    threadRange(0, data.n, data.threadCount, args.id, begin, end);
    data.candidates[args.id] = findLocalPivot(data, begin, end);
    for (std::size_t k = 0; k < data.n; ++k) {
        // При малом числе строк равное деление загружает все доступные потоки.
        const bool useRowQueue = data.threadCount > 1
            && (data.n - k - 1) / data.threadCount >= 2 * eliminationChunkSize;
        finishFactorizationStep(data, k, useRowQueue);
        if (data.status != InversionStatus::success) {
            return nullptr;
        }
        if (data.pivotColumn != k) {
            // Верхняя часть U тоже должна получить перестановку столбцов.
            // Её строки не участвуют в поиске и обновлении активной области.
            threadRange(0, k, data.threadCount, args.id, begin, end);
            for (std::size_t i = begin; i < end; ++i) {
                double* row = data.matrix + i * data.n;
                std::swap(row[k], row[data.pivotColumn]);
            }
        }
        if (useRowQueue) {
            data.candidates[args.id] = eliminateRowChunks(data, k);
        } else {
            threadRange(k + 1, data.n, data.threadCount, args.id, begin, end);
            data.candidates[args.id] = eliminateRows(data, k, begin, end);
        }
    }
    // Последний шаг не обновляет строки. Последний барьер уже опубликовал L,U,P,Q.
    solveInverseColumns(args);
    return nullptr;
}

}

InversionStatus invertMatrixFullPivot(
    std::size_t n,
    std::vector<double>& matrix,
    std::vector<double>& inverse,
    std::vector<std::size_t>& columnPermutation,
    int threadCount
) {
    if (n == 0 || n > std::numeric_limits<std::size_t>::max() / n || threadCount <= 0) {
        return InversionStatus::invalid_input;
    }
    const std::size_t elementCount = n * n;
    if (matrix.size() != elementCount || inverse.size() != elementCount
        || columnPermutation.size() != n || matrix.data() == inverse.data()) {
        return InversionStatus::invalid_input;
    }
    const std::size_t p = std::min(static_cast<std::size_t>(threadCount), n);
    std::vector<std::size_t> rowPermutation(n);
    for (std::size_t i = 0; i < n; ++i) {
        rowPermutation[i] = i;
        columnPermutation[i] = i;
    }
    SharedData data;
    data.n = n;
    data.threadCount = p;
    data.columnTileCount = std::max(p, 1 + (n - 1) / columnTileSize);
    data.matrix = matrix.data();
    data.inverse = inverse.data();
    data.rowPermutation = rowPermutation.data();
    data.columnPermutation = columnPermutation.data();
    data.candidates.resize(p);
    data.remaining.store(p, std::memory_order_relaxed);

    // Внутри pthread-функций нет выделений памяти и исключений.
    std::vector<ThreadArgs> args(p);
    std::vector<pthread_t> threads(p - 1);
    const std::size_t width = 1 + (n - 1) / data.columnTileCount;
    const std::size_t stride = (width + 7) / 8 * 8;
    for (std::size_t t = 0; t < p; ++t) {
        args[t].data = &data;
        args[t].id = t;
        args[t].stride = stride;
        args[t].rightHandSides.resize(n * stride);
    }
    if (p == 1) {
        invertThread(&args[0]);
    } else {
        if (pthread_mutex_init(&data.startMutex, nullptr) != 0) {
            return InversionStatus::thread_failure;
        }
        pthread_mutex_lock(&data.startMutex);
        std::size_t created = 0;
        for (; created < p - 1; ++created) {
            if (pthread_create(&threads[created], nullptr, invertThread, &args[created + 1]) != 0) {
                data.startFailed = true;
                break;
            }
        }
        pthread_mutex_unlock(&data.startMutex);
        if (!data.startFailed) {
            // Вызывающий поток тоже работает: всего вычисляют p потоков.
            invertThread(&args[0]);
        }
        for (std::size_t t = 0; t < created; ++t) {
            pthread_join(threads[t], nullptr);
        }
        pthread_mutex_destroy(&data.startMutex);
        if (data.startFailed) {
            return InversionStatus::thread_failure;
        }
    }
    if (data.status != InversionStatus::success) {
        return data.status;
    }
    for (const ThreadArgs& arg : args) {
        if (arg.failed) {
            return InversionStatus::numerical_failure;
        }
    }
    return InversionStatus::success;
}
