# Timings for the documentation (docs/src/performance.md) and the README: wall time per iteration of the hidden Potts
# sampler, for conditional simulation (20% of the values missing) and unconditional simulation (no data), of 1, 30 and
# 1024 chains on grids of several sizes, on one thread, on all threads and, if CUDA.jl is available and works, on an
# NVIDIA GPU. Threads are used as simulatehiddenpotts uses them: over chains when there are several, and within a chain
# otherwise (for grids large enough for this to pay off). On the GPU, all chains run together, as with
# simulatehiddenpotts(…; backend = CUDABackend()).
#
# The script also times many chains of unconditional simulation with the parameters given per chain, against the same
# parameters shared by all chains: the parameters given per chain equal the shared ones, so that the work is the same,
# and the GPU kernels are made to read them per chain. Finally, it times one call of simulatehiddenpotts that simulates
# a batch of training data, with parameters per chain, on all threads and on the GPU.
#
# Run from the top level of the repository (the first command is needed only once):
#   julia --project=benchmark -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
#   julia --project=benchmark --threads=auto benchmark/table.jl
# For the GPU timings, use the environment benchmark/gpu, which includes CUDA.jl:
#   julia --project=benchmark/gpu -e 'using Pkg; Pkg.instantiate()'
#   julia --project=benchmark/gpu --threads=auto benchmark/table.jl
#
# Every workload is timed in two passes, several minutes apart, and the faster pass is reported: the times of many
# chains on many threads vary between runs, as the chains compete for the memory bandwidth and caches, and occasionally
# by a factor of 2 when another process disturbs them. The script prints a description of the machine, the tables of
# the documentation and of the README as markdown (times in ms to two significant figures, the fastest in bold), the
# extra cost of parameters per chain, and the time to simulate the training data. Every time is also written to
# benchmark/results.csv, from which the documentation draws its figure (for the training data, the time of the whole
# call rather than of one iteration).

using HiddenMarkovRandomFields, Random, Printf
using HiddenMarkovRandomFields: PottsLattice, PottsParams, PottsChain, PottsChains, initialize!, run!
using Distributions: Normal

const HASCUDA = Base.find_package("CUDA") !== nothing
HASCUDA && @eval using CUDA
const GPU = HASCUDA && CUDA.functional()

# On a GPU, chains whose β are all equal share the kernels' parameters, even when they are given per chain. Make the
# kernels read them per chain regardless, to time this.
GPU && (Base.get_extension(HiddenMarkovRandomFields, :HiddenMarkovRandomFieldsKernelAbstractionsExt).SHARED_WHEN_EQUAL[] = false)

const SIZES = [32, 64, 128, 256, 512, 1024]   # grids of SIZES[i] × SIZES[i] pixels
const CHAINS = [1, 30, 1024]                   # numbers of chains
# The largest grid for each number of chains (an iteration of 1024 chains of 512 × 512 pixels takes seconds on one thread)
const MAXSIZE = Dict(1 => 1024, 30 => 1024, 1024 => 256)
const TABLESIZES = [64, 256, 1024]             # grids in the printed tables (results.csv and the figure have them all)
const SIMULATIONS = ("conditional", "unconditional")
const TARGET = 0.5                             # approximate duration of each timed run (seconds)
const β = 1.0                                  # close to the critical value log(1 + √3) ≈ 1.005
const DISTRIBUTIONS = [Normal(-1.0, 0.4), Normal(0.0, 0.4), Normal(1.0, 0.4)]
const T = Threads.nthreads()
const PASSES = 2                               # passes over all the workloads; the fastest is reported
const PERCHAIN = [(64, 1024), (128, 1024), (256, 1024)] # (grid, chains) with parameters shared / given per chain
const TRAINING = (nchains = 1024, n = 64, num_iterations = 200) # the batch of training data

# The n × n data of a simulation: for conditional simulation, an unconditional simulation with 20% of the values then
# removed at random; for unconditional simulation, every value missing
function dataset(n, simulation)
    simulation == "unconditional" && return Matrix{Union{Missing, Float64}}(missing, n, n)
    z = simulatehiddenpotts(n, n, β, DISTRIBUTIONS; num_iterations = 200, rng = Xoshiro(n)).Z[:, :, 1, 1]
    Z = Matrix{Union{Missing, Float64}}(z)
    Z[rand(Xoshiro(n + 1), n, n) .< 0.2] .= missing
    return Z
end

# The parameters of `nchains` chains: one PottsParams shared by all chains, or one per chain (with the same values)
parameters(lattice, nchains, perchain) = perchain ? [PottsParams(lattice, β, DISTRIBUTIONS) for _ = 1:nchains] : PottsParams(lattice, β, DISTRIBUTIONS)
paramsof(params::PottsParams, _) = params
paramsof(params::AbstractVector, c) = params[c]

# Runs `niter` iterations of every chain on `ntasks` tasks, as simulatehiddenpotts(…; nchains = length(chains), ntasks)
# does: the chains are distributed over min(ntasks, nchains) tasks, and any remaining tasks are used within chains
function advance!(chains, lattice, params, niter; sw_every, ntasks)
    outer, inner = HiddenMarkovRandomFields._splittasks(lattice, ntasks, length(chains))
    if outer == 1
        foreach(c -> run!(chains[c], lattice, paramsof(params, c), niter; sw_every = sw_every, ntasks = inner), eachindex(chains))
    else
        Threads.@threads for c in eachindex(chains)
            run!(chains[c], lattice, paramsof(params, c), niter; sw_every = sw_every, ntasks = inner)
        end
    end
end

# Wall time (seconds) per iteration of advance!(niter), excluding initialisation: the number of iterations is chosen so
# that each run takes about TARGET seconds, and the minimum over five runs is reported
function periteration(advance!)
    advance!(1) # compilation (short and long runs of Gibbs sweeps use different kernels on a GPU) and burn-in
    advance!(10)
    niter = 1
    while (t = @elapsed advance!(niter)) < 0.05
        niter *= 2
    end
    niter = max(1, round(Int, niter * TARGET / t))
    return minimum(@elapsed(advance!(niter)) for _ = 1:5) / niter
end

# Time per iteration of `nchains` chains on the CPU, with `ntasks` tasks
function cputime(Z, nchains; sw_every, ntasks, perchain)
    lattice = PottsLattice(Z, DISTRIBUTIONS)
    params = parameters(lattice, nchains, perchain)
    chains = [PottsChain(lattice, Xoshiro(c)) for c = 1:nchains]
    foreach(c -> initialize!(chains[c], lattice, paramsof(params, c)), eachindex(chains))
    return periteration(niter -> advance!(chains, lattice, params, niter; sw_every = sw_every, ntasks = ntasks))
end

# Time per iteration of `nchains` chains on the GPU (run! waits for the GPU to finish)
function gputime(Z, nchains; sw_every, perchain)
    lattice = PottsLattice(Z, DISTRIBUTIONS)
    params = parameters(lattice, nchains, perchain)
    chains = PottsChains(lattice, nchains, CUDA.CUDABackend(); rng = Xoshiro(1))
    initialize!(chains, lattice, params)
    return periteration(niter -> run!(chains, lattice, params, niter; sw_every = sw_every))
end

# Time per iteration of `nchains` chains on a device: one thread (:one), all threads (:all) or the GPU (:gpu)
function devicetime(Z, nchains, device; sw_every, perchain = false)
    device === :gpu && return gputime(Z, nchains; sw_every = sw_every, perchain = perchain)
    return cputime(Z, nchains; sw_every = sw_every, ntasks = device === :one ? 1 : T, perchain = perchain)
end

# Wall time (seconds) of one call of simulatehiddenpotts that simulates a batch of training data, on all threads or on
# the GPU: TRAINING.nchains fields of TRAINING.n × TRAINING.n pixels, each with its own β (uniform on (0, 1.5)) and
# emission standard deviation (uniform on (0.1, 0.6)), after TRAINING.num_iterations iterations of Gibbs + SW. The
# fastest of three calls, after one for compilation.
function trainingtime(device)
    (; nchains, n, num_iterations) = TRAINING
    rng = Xoshiro(1)
    βs = 1.5 .* rand(rng, nchains)
    σs = 0.1 .+ 0.5 .* rand(rng, nchains)
    distributions = [[Normal(-1, σ), Normal(0, σ), Normal(1, σ)] for σ in σs]
    backend = device === :gpu ? CUDA.CUDABackend() : nothing
    function simulate()
        simulatehiddenpotts(n, n, βs, distributions; num_iterations = num_iterations, rng = Xoshiro(2), backend = backend)
        device === :gpu && CUDA.synchronize()
        return nothing
    end
    simulate()
    return minimum(@elapsed(simulate()) for _ = 1:3)
end

# A time in seconds, in milliseconds to two significant figures (e.g., "0.0047", "0.19", "33", "1100")
function msstring(t)
    x = round(1e3t; sigdigits = 2)
    digits = max(0, 1 - floor(Int, log10(x)))
    return Printf.format(Printf.Format("%.$(digits)f"), x)
end

# The times of one workload on several devices, as "a / b / c", with the fastest in bold (and any other time that
# rounds to the same value)
function cellstring(times)
    s = msstring.(times)
    best = s[argmin(times)]
    return join((x == best ? "**$x**" : x for x in s), " / ")
end

const DEVICES = GPU ? (:one, :all, :gpu) : (:one, :all) # one thread, all threads, and the GPU
const MANYDEVICES = GPU ? (:all, :gpu) : (:all,)
devicename(device) = device === :gpu ? "GPU" : device === :one ? "1 thread" : "$T threads"
algorithmname(sw_every) = sw_every == 0 ? "Gibbs only" : "Gibbs + SW"

# The workloads of the main results, as (grid, number of chains)
const WORKLOADS = [(n, nchains) for nchains in CHAINS for n in SIZES if n <= MAXSIZE[nchains]]

# Times per iteration (seconds), the fastest over the passes, and their keys in the order of measurement:
# (table, simulation, parameters, grid, chains, sw_every, device) => time, where the table is "time per iteration"
# (the main results) or "per-chain parameters", and the parameters are "shared" (by all chains) or "per chain"
const Key = Tuple{String, String, String, Int, Int, Int, Symbol}
const TIMES = Dict{Key, Float64}()
const KEYS = Key[]
const TRAININGTIMES = Dict{Symbol, Float64}() # device => time of the whole call (seconds)

function record!(key, t)
    haskey(TIMES, key) || push!(KEYS, key)
    TIMES[key] = min(get(TIMES, key, Inf), t)
    return nothing
end
maintime(simulation, n, nchains, sw_every, device) = TIMES[("time per iteration", simulation, "shared", n, nchains, sw_every, device)]

const DATA = Dict((n, s) => dataset(n, s) for n in SIZES for s in SIMULATIONS)
println("Machine: ", Sys.cpu_info()[1].model, " ($(Sys.CPU_THREADS) logical CPUs), Julia $VERSION, $T threads")
GPU && println("GPU: ", CUDA.name(CUDA.device()), ", CUDA.jl ", pkgversion(CUDA))
println("Hidden Potts model, q = 3, Gaussian emissions, β = $β; conditional simulation (20% of the values missing) and ",
    "unconditional simulation; wall time per iteration in ms (lower is better; the fastest in bold), the fastest of $PASSES passes")

for pass = 1:PASSES
    for (n, nchains) in WORKLOADS, simulation in SIMULATIONS, sw_every in (0, 1), device in DEVICES
        record!(("time per iteration", simulation, "shared", n, nchains, sw_every, device), devicetime(DATA[(n, simulation)], nchains, device; sw_every = sw_every))
    end
    for (n, nchains) in PERCHAIN, sw_every in (0, 1), device in MANYDEVICES, params in ("shared", "per chain")
        t = devicetime(DATA[(n, "unconditional")], nchains, device; sw_every = sw_every, perchain = params == "per chain")
        record!(("per-chain parameters", "unconditional", params, n, nchains, sw_every, device), t)
    end
    for device in MANYDEVICES
        TRAININGTIMES[device] = min(get(TRAININGTIMES, device, Inf), trainingtime(device))
    end
end

println("\nTime per iteration (ms), ", join(devicename.(DEVICES), " / "), " (docs/src/performance.md)\n")
println("| Grid | Chains | Simulation | Gibbs only | Gibbs + SW |\n|:--|--:|:--|--:|--:|")
for n in TABLESIZES
    firstrow = true
    for nchains in CHAINS, (k, simulation) in enumerate(SIMULATIONS)
        n <= MAXSIZE[nchains] || continue
        cells = [cellstring([maintime(simulation, n, nchains, sw_every, device) for device in DEVICES]) for sw_every in (0, 1)]
        println("| ", firstrow ? "$(n)²" : "", " | ", k == 1 ? nchains : "", " | $simulation | ", join(cells, " | "), " |")
        firstrow = false
    end
end

println("\nOne chain, conditional simulation: time per iteration (ms), ", join(devicename.(DEVICES), " / "), " (README.md)\n")
println("| Grid | Gibbs only | Gibbs + SW |\n|:--|--:|--:|")
for n in TABLESIZES
    cells = [cellstring([maintime("conditional", n, 1, sw_every, device) for device in DEVICES]) for sw_every in (0, 1)]
    println("| $n × $n | ", join(cells, " | "), " |")
end

println("\nParameters per chain, unconditional simulation: time per iteration (ms), shared / per chain (ratio)\n")
for (n, nchains) in PERCHAIN, sw_every in (0, 1), device in MANYDEVICES
    shared, perchain = (TIMES[("per-chain parameters", "unconditional", p, n, nchains, sw_every, device)] for p in ("shared", "per chain"))
    println("$nchains chains of $n × $n, $(algorithmname(sw_every)), $(devicename(device)): ", msstring(shared), " / ", msstring(perchain),
        " (", round(perchain / shared; digits = 3), ")")
end

println("\nTraining data: one call of simulatehiddenpotts for $(TRAINING.nchains) fields of $(TRAINING.n) × $(TRAINING.n) pixels, ",
    "each with its own β and emission distributions, $(TRAINING.num_iterations) iterations: ",
    join(("$(round(TRAININGTIMES[device]; sigdigits = 2)) s on $(devicename(device))" for device in MANYDEVICES), ", "))

open(joinpath(@__DIR__, "results.csv"), "w") do io
    println(io, "table,simulation,parameters,grid,chains,algorithm,device,ms")
    for key in KEYS
        table, simulation, params, n, nchains, sw_every, device = key
        println(io, join((table, simulation, params, n, nchains, algorithmname(sw_every), devicename(device), @sprintf("%.6g", 1e3TIMES[key])), ","))
    end
    for device in MANYDEVICES
        println(io, join(("training data", "unconditional", "per chain", TRAINING.n, TRAINING.nchains, "Gibbs + SW", devicename(device),
            @sprintf("%.6g", 1e3TRAININGTIMES[device])), ","))
    end
end
println("\nWrote every time (ms) to ", joinpath(@__DIR__, "results.csv"))
