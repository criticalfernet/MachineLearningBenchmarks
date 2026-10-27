import numpy as np
from pathlib import Path
import csv
import time
from threadpoolctl import threadpool_limits

from matrixMul import multiply

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "datasets" / "matrixMultiplication"
RESULT_FILE = ROOT / "results" / "matrixMultiplication_python.csv"

WARMUPS = 10
REPETITIONS = 20

def threads_for(n):
  return 8 if n >= 768 else (4 if n >= 384 else 1)



def load_test(filePath, MATRIX_SIZE):
  line = filePath.readline()
  while line and not line.startswith("# Test"):
    line = filePath.readline()
  if not line:
    return None

  matrices = []
  for _ in range(3):
    filePath.readline()

    matrix = np.array([
      [float(x) for x in filePath.readline().strip().split(',')]
      for _ in range(MATRIX_SIZE)
    ])

    matrices.append(matrix)

  A, B, C = matrices
  return (A,B,C)



def load_dataset(path):
  print(f"Loading Dataset: {path.name}")
  tests = []

  with open(path, "r") as f:
    num_tests = int(f.readline())
    n = int(f.readline())

    for _ in range(num_tests):
      res = load_test(f, n)
      if res == None:
        break
      tests.append(res)

  return tests



def save_results(test_results, summary, MATRIX_SIZE):
    with RESULT_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "implementation",
            "algorithm",
            "matrix_size",
            "test_case",
            "median_latency_ms",
            "mean_latency_ms",
            "min_latency_ms",
            "max_latency_ms",
            "warmup_runs",
            "repetitions",
        ])

        for row in test_results:
            writer.writerow([
                "Python",
                "Matrix Multiplication",
                MATRIX_SIZE,
                row["test_case"],
                f'{row["median_ms"]:.6f}',
                f'{row["mean_ms"]:.6f}',
                f'{row["min_ms"]:.6f}',
                f'{row["max_ms"]:.6f}',
                WARMUPS,
                REPETITIONS,
            ])

        writer.writerow([
            "Python",
            "Matrix Multiplication",
            MATRIX_SIZE,
            "SUMMARY",
            f'{summary["median_ms"]:.6f}',
            f'{summary["mean_ms"]:.6f}',
            f'{summary["min_ms"]:.6f}',
            f'{summary["max_ms"]:.6f}',
            WARMUPS,
            REPETITIONS,
        ])

    print(f"\nResults saved to: {RESULT_FILE}")



def benchmark(dataset):
  tests = load_dataset(dataset)
  if not tests:
    print(f"No test cases found in {dataset}")
    return
  
  n = tests[0][0].shape[0]
  timings = []
  test_results = []

  print(f"Test cases in dataset: {len(tests)}")

  for idx, (A, B, C) in enumerate(tests, start=1):
    # with threadpool_limits(limits=threads_for(n), user_api="blas"):
    for _ in range(WARMUPS):
      multiply(A,B)

    samples = []

    for _ in range(REPETITIONS):
      start = time.perf_counter()
      result = multiply(A,B)
      end = time.perf_counter()
      samples.append(end-start)

    if not np.allclose(result, C, rtol=1e-10, atol=1e-10):
      raise ValueError(f"Test {idx}: correctness check failed")

    median_time = float(np.median(samples))
    timings.append(median_time)

    test_results.append({
      "test_case": idx,
      "median_ms": median_time * 1000,
      "mean_ms": float(np.mean(samples)) * 1000,
      "min_ms": float(np.min(samples)) * 1000,
      "max_ms": float(np.max(samples)) * 1000,
    })

    print(f"Test {idx}: done")

  timings = np.array(timings)
  median = float(np.median(timings))
  mean = float(np.mean(timings))
  min = float(np.min(timings))
  max = float(np.max(timings))
  summary = {
    "median_ms": median * 1000,
    "mean_ms": mean * 1000,
    "min_ms": min * 1000,
    "max_ms": max * 1000,
  }

  print("\n--- Benchmark Completed ---")
  save_results(test_results, summary,n)



if __name__ == "__main__":
  datasets = sorted(DATASET_DIR.glob("matrix_*.txt"),
    key=lambda path: int(path.stem.removeprefix("matrix_"))            
  )

  if not datasets:
    raise FileNotFoundError(f"No matrix_*.txt datasets found in {DATASET_DIR}")

  for dataset in datasets:
    benchmark(dataset)