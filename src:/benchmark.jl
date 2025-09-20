# src/benchmark.jl

using ArgParse, DataFrames, CSV, BenchmarkTools, Distributed

# Ensure other scripts can be included
cd(dirname(@__DIR__))
include("src/montecarlo.jl")
include("src/parallel_threads.jl")
include("src/parallel_distributed.jl")

function parse_benchmark_args()
    s = ArgParseSettings(description="Benchmark Runner for Monte Carlo Simulations")
    @add_arg_table! s begin
        "--npaths"
            help = "Number of simulation paths"
            arg_type = Int
            default = 1_000_000
        "--nsteps"
            help = "Number of time steps per path"
            arg_type = Int
            default = 252
        "--output", "-o"
            help = "Output CSV file path"
            arg_type = String
            default = "results/benchmark_summary.csv"
        "--seed"
            help = "Random seed"
            arg_type = Int
            default = 42
    end
    return parse_args(s)
end

function run_python_benchmark(n_paths, n_steps, seed, impl)
    cmd = `python3 python/reference_mc.py --npaths $n_paths --nsteps $n_steps --seed $seed --impl $impl`
    output = read(cmd, String)
    
    # Parse the machine-readable line
    for line in split(output, '\n')
        if startswith(line, "CSV_OUTPUT")
            parts = split(line, ',')
            time_sec = parse(Float64, parts[2])
            var = parse(Float64, parts[3])
            es = parse(Float64, parts[4])
            return (time_sec=time_sec, var=var, es=es)
        end
    end
    error("Could not parse Python script output.")
end

function main_benchmark()
    args = parse_benchmark_args()
    
    const S0 = 100.0
    const μ = 0.05
    const σ = 0.20
    const T = 1.0
    const α = 0.99
    
    results = DataFrame(
        implementation=String[],
        n_paths=Int[],
        n_steps=Int[],
        n_threads=Int[],
        n_procs=Int[],
        time_sec=Float64[],
        mem_bytes=Int[],
        var=Float64[],
        es=Float64[]
    )
    
    # --- Serial Julia ---
    println("\nBenchmarking: Julia Serial")
    b = @benchmark run_serial_mc($args["npaths"], $args["nsteps"], $S0, $μ, $σ, $T, $α, $args["seed"])
    risks = run_serial_mc(args["npaths"], args["nsteps"], S0, μ, σ, T, α, args["seed"])
    push!(results, ("julia_serial", args["npaths"], args["nsteps"], 1, 1, median(b.times) / 1e9, b.memory, risks.VaR, risks.ES))

    # --- Threaded Julia ---
    max_threads = Threads.nthreads()
    for nt in 1:max_threads
        # Set a dummy task to ensure all threads are active for benchmark
        Threads.@threads for i in 1:nt end
        println("Benchmarking: Julia Threads (n=$nt)")
        b = @benchmark run_threads_mc($args["npaths"], $args["nsteps"], $S0, $μ, $σ, $T, $α, $args["seed"])
        risks = run_threads_mc(args["npaths"], args["nsteps"], S0, μ, σ, T, α, args["seed"])
        push!(results, ("julia_threads", args["npaths"], args["nsteps"], nt, 1, median(b.times) / 1e9, b.memory, risks.VaR, risks.ES))
    end
    
    # --- Distributed Julia ---
    max_procs = nprocs()
    if max_procs > 1
        println("Benchmarking: Julia Distributed (n=$(nworkers()))")
        b = @benchmark run_distributed_mc($args["npaths"], $args["nsteps"], $S0, $μ, $σ, $T, $α, $args["seed"])
        risks = run_distributed_mc(args["npaths"], args["nsteps"], S0, μ, σ, T, α, args["seed"])
        push!(results, ("julia_distributed", args["npaths"], args["nsteps"], 1, nworkers(), median(b.times) / 1e9, b.memory, risks.VaR, risks.ES))
    end

    # --- Python Numpy ---
    println("Benchmarking: Python Numpy")
    py_res_np = run_python_benchmark(args["npaths"], args["nsteps"], args["seed"], "numpy")
    push!(results, ("python_numpy_serial", args["npaths"], args["nsteps"], 1, 1, py_res_np.time_sec, 0, py_res_np.var, py_res_np.es))

    # --- Python Numba ---
    println("Benchmarking: Python Numba")
    py_res_nb = run_python_benchmark(args["npaths"], args["nsteps"], args["seed"], "numba")
    push!(results, ("python_numba_serial", args["npaths"], args["nsteps"], 1, 1, py_res_nb.time_sec, 0, py_res_nb.var, py_res_nb.es))
    
    # --- Post-processing ---
    baseline_time = results[results.implementation .== "julia_serial", :time_sec][1]
    results.paths_per_sec = results.n_paths ./ results.time_sec
    results.speedup = baseline_time ./ results.time_sec
    
    # Save results
    output_path = args["output"]
    mkpath(dirname(output_path))
    CSV.write(output_path, results)
    
    println("\nBenchmark summary saved to $output_path")
    println(results)
end

main_benchmark()