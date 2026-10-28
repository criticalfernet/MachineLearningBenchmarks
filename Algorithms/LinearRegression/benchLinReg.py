import numpy as np
import pandas as pd
from pathlib import Path
import time
import csv

from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
DATASET_FILE = ROOT / "datasets" / "LinearRegression" / "train.csv"
RESULT_FILE = ROOT / "results" / "LinearRegression_python.csv"

WARMUPS = 3
REPETITIONS = 10

from linReg import linear_regression


def save_results(summary, SIZE):
  with RESULT_FILE.open("w", newline="", encoding="utf-8") as f:
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
      writer.writerow([
        "Python",
        "Linear Regression",
        f"{SIZE['x']}x{SIZE['y']}",
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
    X, Y = dataset
    print(f"Dataset shape: X={X.shape}, Y={Y.shape}")
    timings = []

    with threadpool_limits(limits=8, user_api="blas"):
      for _ in range(WARMUPS):
          linear_regression(X, Y)

      for _ in range(REPETITIONS):
          start = time.perf_counter()
          linear_regression(X, Y)
          end = time.perf_counter()
          timings.append(end - start)

    timings = np.array(timings)

    summary = {
        "median_ms": float(np.median(timings)) * 1000,
        "mean_ms": float(np.mean(timings)) * 1000,
        "min_ms": float(np.min(timings)) * 1000,
        "max_ms": float(np.max(timings)) * 1000,
    }

    print("\n--- Benchmark Completed ---")
    save_results(summary, {"x":X.shape[0], "y":X.shape[1]})


if __name__ == "__main__":
  dataset = pd.read_csv(DATASET_FILE, header=None)

  x = dataset.iloc[:, :512].to_numpy(dtype=np.float64)
  y = dataset.iloc[:, 512].to_numpy(dtype=np.float64)

  benchmark((x,y))