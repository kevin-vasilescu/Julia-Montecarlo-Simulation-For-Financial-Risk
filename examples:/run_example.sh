#!/bin/bash
# examples/run_example.sh

# This script provides example commands to run the Monte Carlo simulations.
# Uncomment the line you wish to execute.

# --- Basic Parameters ---
NPATHS=1000000
NSTEPS=252
ALPHA=0.99
SEED=42

echo "================================================="
echo "Running Julia Serial Implementation"
echo "================================================="
# julia --project=. src/montecarlo.jl --npaths $NPATHS --nsteps $NSTEPS --alpha $ALPHA --seed $SEED


echo "================================================="
echo "Running Julia Multi-Threaded (4 Threads)"
echo "================================================="
# export JULIA_NUM_THREADS=4
# julia --project=. src/parallel_threads.jl --npaths $NPATHS --nsteps $NSTEPS --alpha $ALPHA --seed $SEED


echo "================================================="
echo "Running Julia Multi-Process (4 Workers)"
echo "================================================="
# The main process counts as 1, so -p 4 gives you 3 workers. Use -p 5 for 4 workers.
# julia --project=. -p 5 src/parallel_distributed.jl --npaths $NPATHS --nsteps $NSTEPS --alpha $ALPHA --seed $SEED


echo "================================================="
echo "Running Python Numpy Reference"
echo "================================================="
# python3 python/reference_mc.py --npaths $NPATHS --nsteps $NSTEPS --alpha $ALPHA --seed $SEED


echo "================================================="
echo "Running Python Numba Reference"
echo "================================================="
# python3 python/reference_mc.py --npaths $NPATHS --nsteps $NSTEPS --alpha $ALPHA --seed $SEED --impl numba


echo "================================================="
echo "Running the Full Benchmark Suite (8 cores)"
echo "================================================="
# This will run all implementations and save a CSV summary.
# It assumes you have 8 cores available for threads and processes.
export JULIA_NUM_THREADS=8
julia --project=. -p 9 src/benchmark.jl --npaths $NPATHS --output results/benchmark_summary_$(date +%F).csv

echo "Done. Check the output above and the generated CSV in the 'results' directory."