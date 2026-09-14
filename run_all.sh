#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export PYTHONPATH=.
mkdir -p results figures
python3 tests/test_all.py
python3 benchmarks/bench_solvers.py --out results/solvers.csv
python3 benchmarks/bench_blocksize.py --out results/blocksize.csv --trials 120
python3 benchmarks/bench_threshold.py --out results/threshold.csv --reps 3
python3 benchmarks/bench_mayo.py --out results/mayo.csv --reps 5
python3 benchmarks/bench_robust.py --out results/robust.csv --reps 30
python3 benchmarks/make_figures.py --results results --out figures
