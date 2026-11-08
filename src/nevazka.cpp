#include "nevazka.hpp"

#include <pthread.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

struct ResidualArgs {
    std::size_t n = 0;
    std::size_t firstRow = 0;
    std::size_t lastRow = 0;
    const double* original = nullptr;
    const double* inverse = nullptr;
    std::vector<double> productRow;
    double norm = 0.0;
    bool failed = false;
};

void* residualThread(void* argument) {
    ResidualArgs& args = *static_cast<ResidualArgs*>(argument);
    const std::size_t n = args.n;
    std::vector<double>& productRow = args.productRow;

    for (std::size_t i = args.firstRow; i < args.lastRow; ++i) {
        const double* originalRow = args.original + i * n;
        std::fill(productRow.begin(), productRow.end(), 0.0);

        for (std::size_t k = 0; k < n; ++k) {
            const double coefficient = originalRow[k];
            const double* inverseRow = args.inverse + k * n;
            for (std::size_t j = 0; j < n; ++j) {
                productRow[j] += coefficient * inverseRow[j];
            }
        }

        double rowSum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            if (!std::isfinite(productRow[j])) {
                args.failed = true;
                return nullptr;
            }

            const double residual = productRow[j] - (i == j ? 1.0 : 0.0);
            if (!std::isfinite(residual)) {
                args.failed = true;
                return nullptr;
            }
            rowSum += std::abs(residual);
            if (!std::isfinite(rowSum)) {
                args.failed = true;
                return nullptr;
            }
        }
        if (rowSum > args.norm) {
            args.norm = rowSum;
        }
    }
    return nullptr;
}

}

bool residualInfinityNorm(
    std::size_t n,
    const std::vector<double>& original,
    const std::vector<double>& inverse,
    int threadCount,
    double& norm
) {
    if (n == 0 || n > std::numeric_limits<std::size_t>::max() / n || threadCount <= 0) {
        return false;
    }

    const std::size_t elementCount = n * n;
    if (original.size() != elementCount || inverse.size() != elementCount) {
        return false;
    }

    const std::size_t p = static_cast<std::size_t>(threadCount);
    std::vector<ResidualArgs> args(p);
    std::vector<pthread_t> threads(p);

    for (std::size_t t = 0; t < p; ++t) {
        args[t].n = n;
        args[t].firstRow = n * t / p;
        args[t].lastRow = n * (t + 1) / p;
        args[t].original = original.data();
        args[t].inverse = inverse.data();
        args[t].productRow.assign(n, 0.0);
    }

    std::size_t created = 0;
    bool startFailed = false;
    for (; created < p; ++created) {
        if (pthread_create(&threads[created], nullptr, residualThread, &args[created]) != 0) {
            startFailed = true;
            break;
        }
    }

    for (std::size_t t = 0; t < created; ++t) {
        pthread_join(threads[t], nullptr);
    }
    if (startFailed) {
        return false;
    }

    norm = 0.0;
    for (const ResidualArgs& current : args) {
        if (current.failed) {
            return false;
        }
        norm = std::max(norm, current.norm);
    }
    return true;
}
