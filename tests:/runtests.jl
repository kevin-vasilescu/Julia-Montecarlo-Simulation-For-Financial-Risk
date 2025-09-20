# tests/runtests.jl

using Test, Statistics
# Make sure we are in the project directory so includes work
cd(dirname(@__DIR__))

include("src/montecarlo.jl")
include("src/parallel_threads.jl")
include("src/parallel_distributed.jl")

# Test parameters
const N_TEST_PATHS = 10_000
const N_TEST_STEPS = 50
const S0_TEST = 100.0
const μ_TEST = 0.05
const σ_TEST = 0.20
const T_TEST = 1.0
const α_TEST = 0.95
const SEED_TEST = 123

@testset "Monte Carlo Risk Analysis Tests" begin

    @testset "Correctness and Reproducibility" begin
        # Run serial twice with same seed
        risks1 = run_serial_mc(N_TEST_PATHS, N_TEST_STEPS, S0_TEST, μ_TEST, σ_TEST, T_TEST, α_TEST, SEED_TEST)
        risks2 = run_serial_mc(N_TEST_PATHS, N_TEST_STEPS, S0_TEST, μ_TEST, σ_TEST, T_TEST, α_TEST, SEED_TEST)

        # Should be exactly the same
        @test risks1.VaR == risks2.VaR
        @test risks1.ES == risks2.ES
        
        # Plausibility check
        @test 0.0 < risks1.VaR < 0.5
        @test risks1.VaR < risks1.ES < 0.6
    end

    @testset "Consistency Across Implementations" begin
        # Use a higher tolerance because different parallel RNG streams lead to statistical variation
        TOLERANCE = 0.05 # 5% tolerance

        risks_serial = run_serial_mc(N_TEST_PATHS, N_TEST_STEPS, S0_TEST, μ_TEST, σ_TEST, T_TEST, α_TEST, SEED_TEST)
        
        # Test Threads
        risks_threads = run_threads_mc(N_TEST_PATHS, N_TEST_STEPS, S0_TEST, μ_TEST, σ_TEST, T_TEST, α_TEST, SEED_TEST)
        @test isapprox(risks_serial.VaR, risks_threads.VaR, rtol=TOLERANCE)
        @test isapprox(risks_serial.ES, risks_threads.ES, rtol=TOLERANCE)

        # Test Distributed
        # Add a worker process programmatically for testing if none exist
        if nworkers() == 0
            addprocs(1)
            # Make sure the new process has the necessary code
            @everywhere include("src/montecarlo.jl")
            @everywhere @everywhere function run_chunk(chunk_params::NamedTuple)::Vector{Float64}
                n_chunk_paths, n_steps, S0, μ, σ, T, seed = chunk_params
                rng = MersenneTwister(seed)
                chunk_prices = zeros(Float64, n_chunk_paths)
                for i in 1:n_chunk_paths
                    chunk_prices[i] = simulate_gbm_path(S0, μ, σ, T, n_steps, rng)
                end
                return chunk_prices
            end
        end

        risks_dist = run_distributed_mc(N_TEST_PATHS, N_TEST_STEPS, S0_TEST, μ_TEST, σ_TEST, T_TEST, α_TEST, SEED_TEST)
        @test isapprox(risks_serial.VaR, risks_dist.VaR, rtol=TOLERANCE)
        @test isapprox(risks_serial.ES, risks_dist.ES, rtol=TOLERANCE)
    end
end

println("All tests passed!")