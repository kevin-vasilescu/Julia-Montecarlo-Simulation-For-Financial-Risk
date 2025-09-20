# Parallel Monte Carlo Risk Analysis in Julia

This project demonstrates high-performance parallel computing in Julia by implementing Monte Carlo simulations to calculate financial risk metrics: **Value at Risk (VaR)** and **Expected Shortfall (ES)**.

It includes:
- A baseline **serial** implementation.
- A **multi-threaded** implementation using `Threads.@threads`.
- A **multi-process (distributed)** implementation using `Distributed` and `pmap`.
- A reference **Python** implementation (`numpy` and `numba`) for benchmarking.
- Unit tests, a benchmarking script, and result analysis.

---

## Table of Contents
1.  [Project Structure](#project-structure)
2.  [Installation](#installation)
3.  [How to Run](#how-to-run)
    - [Single Simulation Runs](#single-simulation-runs)
    - [Running Benchmarks](#running-benchmarks)
    - [Running Tests](#running-tests)
4.  [Parallelization Strategy](#parallelization-strategy)
    - [Multi-Threading (`Threads`)](#multi-threading-threads)
    - [Multi-Processing (`Distributed`)](#multi-processing-distributed)
5.  [Benchmark Results](#benchmark-results)
6.  [Analysis](#analysis)

---

## Project Structure

- `src/`: Contains the core Julia simulation logic.
  - `montecarlo.jl`: Serial GBM simulator and risk metric calculations.
  - `parallel_threads.jl`: Multi-threaded implementation.
  - `parallel_distributed.jl`: Multi-process (distributed) implementation.
- `python/`: Contains the Python reference implementation for comparison.
- `tests/`: Unit and integration tests.
- `examples/`: Shell script with sample run commands.
- `results/`: Default output directory for benchmarks.
- `notebooks/`: Contains a markdown file with analysis and plotting instructions.
- `Project.toml`: Julia project dependencies.

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install Julia dependencies:**
    Launch the Julia REPL in the project directory and run the following commands.
    ```bash
    julia --project=.
    ```
    In the Julia package manager prompt (`pkg>`), run:
    ```julia
    ] activate .
    ] instantiate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install numpy numba pandas
    ```

---

## How to Run

### Single Simulation Runs

You can run a single simulation using one of the main scripts in `src/`. All scripts share a common set of command-line arguments.

**Arguments:**
- `--npaths`: Number of simulation paths (e.g., 1000000).
- `--nsteps`: Number of time steps per path (e.g., 252).
- `--alpha`: Confidence level for VaR/ES (e.g., 0.99).
- `--seed`: Random number generator seed for reproducibility.

**1. Serial Julia**
```bash
julia --project=. src/montecarlo.jl --npaths 1000000 --nsteps 252 --alpha 0.99