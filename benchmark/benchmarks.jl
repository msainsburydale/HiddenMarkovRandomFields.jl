# Benchmarks of the Potts samplers, run on every pull request by AirspeedVelocity.jl (.github/workflows/benchmark_pr.yml).
#
# To run them locally, from the top level of the repository:
#   julia --project=benchmark -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
#   julia --project=benchmark --threads=auto -e 'include("benchmark/benchmarks.jl"); run(SUITE; verbose = true)'

using BenchmarkTools, HiddenMarkovRandomFields, Random
using Distributions: Normal, Beta, Dirac

const SUITE = BenchmarkGroup()
SUITE["hidden Potts"] = BenchmarkGroup()
SUITE["Potts"] = BenchmarkGroup()

# Incomplete 64×64 data set (q = 3, Normal emissions, 20% missing completely at random), as in one neural EM run
distributions = [Normal(-1.0, 0.3), Normal(0.0, 0.3), Normal(1.0, 0.3)]
β = 1.0
Z = simulatehiddenpotts(64, 64, β, distributions; num_iterations = 100, rng = Xoshiro(1)).Z[:, :, 1, 1]
Z₁ = Matrix{Union{Missing, Float64}}(Z)
Z₁[rand(Xoshiro(2), 64, 64) .< 0.2] .= missing

for (name, sw_every) in (("Gibbs", 0), ("Gibbs + SW", 1))
    SUITE["hidden Potts"]["conditional 64×64, 30 chains × 50 iterations, $name"] =
        @benchmarkable simulatehiddenpotts($Z₁, $β, $distributions; nchains = 30, num_iterations = 50, sw_every = $sw_every, rng = Xoshiro(3))
end

SUITE["hidden Potts"]["unconditional 100×100, 1000 iterations"] =
    @benchmarkable simulatehiddenpotts(100, 100, $β, $distributions; num_iterations = 1000, rng = Xoshiro(4))

# One chain on a large grid, with its sweeps parallelised over all threads
Zl = simulatehiddenpotts(512, 512, β, distributions; num_iterations = 50, rng = Xoshiro(9)).Z[:, :, 1, 1]
Zl₁ = Matrix{Union{Missing, Float64}}(Zl)
Zl₁[rand(Xoshiro(10), 512, 512) .< 0.2] .= missing

SUITE["hidden Potts"]["conditional 512×512, one chain × 20 iterations, all threads"] =
    @benchmarkable simulatehiddenpotts($Zl₁, $β, $distributions; nchains = 1, num_iterations = 20, rng = Xoshiro(11))

# Sea-ice style data: Beta and Dirac emissions, with a large missing block
seaice = [Beta(2.0, 5.0), Beta(5.0, 2.0), Dirac(0.0), Dirac(1.0)]
Zs = simulatehiddenpotts(100, 110, 1.2, seaice; num_iterations = 200, rng = Xoshiro(5)).Z[:, :, 1, 1]
Zs₁ = Matrix{Union{Missing, Float64}}(Zs)
Zs₁[rand(Xoshiro(6), 100, 110) .< 0.1] .= missing
Zs₁[30:60, 40:80] .= missing

SUITE["hidden Potts"]["conditional 100×110, Beta + Dirac, 10 chains × 100 iterations"] =
    @benchmarkable simulatehiddenpotts($Zs₁, 1.2, $seaice; nchains = 10, num_iterations = 100, rng = Xoshiro(7))

SUITE["Potts"]["unconditional 100×100, q = 3, 1000 iterations"] =
    @benchmarkable simulatepotts(100, 100, 3, 1.0; num_iterations = 1000, rng = Xoshiro(8))
