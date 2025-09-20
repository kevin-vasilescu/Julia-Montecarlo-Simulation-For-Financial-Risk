# src/parallel_threads.jl

using Base.Threads, Random, ArgParse

# Include the core serial logic
include("montecarlo.jl")

"""
    run_threads_mc(n_paths, n_steps, S0, μ, σ, T, α, seed)

Runs a Monte Carlo simulation using multiple threads.
"""
function run_threads_mc(n_paths::Int, n_steps::Int, S0::Float64, μ::Float64, σ::Float64, T::Float64, α::Float64, seed::Int)
    final_prices = zeros(Float64, n_paths)
    
    # Create a thread-safe RNG for each thread
    rngs = [MersenneTwister(seed + i) for i in 1:nthreads()]
    
    # The @threads macro partitions the loop iterations among available threads.
    # Each thread gets its own RNG and writes to a unique location in the final_prices array.
    # This is a data-parallel pattern with no race conditions.
    @threads for i in 1:n_paths
        thread_id = threadid()
        final_prices[i] = simulate_gbm_path(S0, μ, σ, T, n_steps, rngs[thread_id])
    end
    
    risks = calculate_risks(S0, final_prices, α)
    return risks
end

function main_threads()
    args = parse_commandline()
    
    const S0 = 100.0
    const μ = 0.05
    const σ = 0.20
    const T = 1.0
    
    num_threads = nthreads()
    if num_threads == 1
        @warn "Running on a single thread. Set JULIA_NUM_THREADS > 1 for parallel execution."
    end
    
    println("Running Multi-Threaded Monte Carlo with $num_threads threads...")
    println("Config: n_paths=$(args["npaths"]), n_steps=$(args["nsteps"]), alpha=$(args["alpha"])")

    # Warm-up run
    run_threads_mc(100, 10, S0, μ, σ, T, args["alpha"], args["seed"])
    
    # Timed run
    elapsed_time = @elapsed begin
        risks = run_threads_mc(args["npaths"], args["nsteps"], S0, μ, σ, T, args["alpha"], args["seed"])
        println("Results: VaR = $(round(risks.VaR, digits=4)), ES = $(round(risks.ES, digits=4))")
    end
    
    paths_per_sec = args["npaths"] / elapsed_time
    println("Completed in $(round(elapsed_time, digits=2)) seconds ($(round(paths_per_sec, digits=0)) paths/sec).")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_threads()
end