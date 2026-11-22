"""Compare inversion wall times reported by program; input/output and residual are excluded."""

import argparse
import json
import random
import statistics
import subprocess
import tempfile
from pathlib import Path


def main():
    # Собираем набор запусков для указанных размеров, версий и числа потоков.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--matrix", choices=["1", "2", "3", "random"], default="1")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 for n in args.sizes + args.threads):
        parser.error("sizes, threads and repeats must be positive")
    versions = {"new": args.program.resolve()}
    if args.baseline:
        versions["baseline"] = args.baseline.resolve()
    results = []
    # Один и тот же входной файл используется всеми сравниваемыми версиями.
    with tempfile.TemporaryDirectory(prefix="gauss-benchmark-") as directory:
        for n in args.sizes:
            matrix_args = [args.matrix]
            if args.matrix == "random":
                # Строго диагонально преобладающая матрица с независимыми
                # перестановками строк и столбцов для проверки работы с перестановками.
                generator = random.Random(n)
                rows, columns = list(range(n)), list(range(n))
                generator.shuffle(rows)
                generator.shuffle(columns)
                matrix_path = Path(directory) / "matrix.txt"
                with matrix_path.open("w", encoding="ascii") as output:
                    for row in rows:
                        output.write(" ".join(
                            format(generator.uniform(-1, 1) + (n + 1 if row == column else 0), ".17g")
                            for column in columns) + "\n")
                matrix_args = ["0", str(matrix_path)]
            cases = [(version, p) for version in versions for p in args.threads]
            samples = {case: [] for case in cases}
            residuals = {case: [] for case in cases}
            for repeat in range(args.repeats):
                # Чередование версий и числа потоков уменьшает влияние порядка запусков.
                order = cases[repeat % len(cases):] + cases[:repeat % len(cases)]
                for version, p in order:
                    command = [str(versions[version]), str(n), "0", str(p), *matrix_args]
                    run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                         check=True, timeout=600)
                    # Программа печатает невязку и время обращения в последних двух строках.
                    lines = run.stdout.splitlines()
                    seconds = float(lines[-1].split()[-2])
                    residual = float(lines[-2].rsplit(" ", 1)[-1])
                    if not (seconds > 0 and 0 <= residual < 1e-6):
                        raise RuntimeError(f"invalid timing or residual: {command}: {lines[-2:]}")
                    samples[version, p].append(seconds)
                    residuals[version, p].append(residual)
                    print(f"sample {version:8} n={n} p={p} t={seconds:.6f}s "
                          f"residual={residual:.3e}", flush=True)
            # Ускорение считаем относительно одного потока той же версии программы.
            for version, p in cases:
                median = statistics.median(samples[version, p])
                one_thread = statistics.median(samples[version, 1]) if 1 in args.threads else None
                result = dict(version=version, n=n, p=p, matrix=args.matrix,
                              median_seconds=median, samples=samples[version, p],
                              residual_max=max(residuals[version, p]))
                if one_thread is not None:
                    result.update(speedup=one_thread / median, efficiency=one_thread / median / p)
                results.append(result)
                print("result " + json.dumps(result), flush=True)
    # Сохраняем и медианы, и отдельные измерения для проверки разброса.
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
