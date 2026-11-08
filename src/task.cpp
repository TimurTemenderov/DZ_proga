#include "task.hpp"

#include <pthread.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

constexpr std::size_t stepBlockSize = 64;
constexpr std::size_t columnTileSize = 128;

struct PivotCandidate {
    double magnitude = 0.0;
    std::size_t row = 0;
    std::size_t column = 0;
    bool failed = false;
};

struct SharedData {
    std::size_t n = 0;
    std::size_t threadCount = 0;
    double* matrix = nullptr;
    double* inverse = nullptr;
    std::size_t* columnPermutation = nullptr;
    double tolerance = 0.0;

    pthread_barrier_t barrier;
    pthread_mutex_t startMutex;
    bool startFailed = false;

    std::vector<PivotCandidate> candidates;
    InversionStatus status = InversionStatus::success;
};

struct ThreadArgs {
    SharedData* data = nullptr;
    std::size_t id = 0;
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

void backSubstitution(const SharedData& data, std::size_t begin, std::size_t end) {
    const std::size_t n = data.n;
    for (std::size_t tileBegin = begin; tileBegin < end; tileBegin += columnTileSize) {
        const std::size_t tileEnd = std::min(tileBegin + columnTileSize, end);
        for (std::size_t ii = n; ii-- > 0;) {
            double* inverseRow = data.inverse + ii * n;
            const double* row = data.matrix + ii * n;
            for (std::size_t r = ii + 1; r < n; ++r) {
                const double coefficient = row[r];
                const double* solvedRow = data.inverse + r * n;
                for (std::size_t j = tileBegin; j < tileEnd; ++j) {
                    inverseRow[j] -= coefficient * solvedRow[j];
                }
            }

            const double invDiagonal = 1.0 / row[ii];
            for (std::size_t j = tileBegin; j < tileEnd; ++j) {
                inverseRow[j] *= invDiagonal;
            }
        }
    }
}

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

    std::size_t begin = 0;
    std::size_t end = 0;
    threadRange(0, n, data.threadCount, id, begin, end);
    data.candidates[id] = findLocalPivot(data, begin, end);

    const std::size_t columnBegin = begin;
    const std::size_t columnEnd = end;

    for (std::size_t k = 0; k < n; ++k) {
        pthread_barrier_wait(&data.barrier);
        if (k > 0 && k % stepBlockSize == 0) {
            applyDelayedSteps(data, k - stepBlockSize, k, columnBegin, columnEnd);
            pthread_barrier_wait(&data.barrier);
        }
        if (id == 0) {
            selectPivot(data, k);
        }
        pthread_barrier_wait(&data.barrier);
        if (data.status != InversionStatus::success) {
            return nullptr;
        }

        threadRange(k + 1, n, data.threadCount, id, begin, end);
        data.candidates[id] = eliminateRows(data, k, begin, end);
    }

    applyDelayedSteps(data, (n - 1) / stepBlockSize * stepBlockSize, n, columnBegin, columnEnd);
    backSubstitution(data, columnBegin, columnEnd);
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
    if (matrix.size() != elementCount || inverse.size() != elementCount || columnPermutation.size() != n) {
        return InversionStatus::invalid_input;
    }

    double scale = 0.0;
    for (const double value : matrix) {
        if (!std::isfinite(value)) {
            return InversionStatus::numerical_failure;
        }
        scale = std::max(scale, std::abs(value));
    }

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

    restoreRowOrder(n, inverse, columnPermutation);
    for (const double value : inverse) {
        if (!std::isfinite(value)) {
            return InversionStatus::numerical_failure;
        }
    }
    return InversionStatus::success;
}
