# Examples

The examples use [CairoMakie.jl](https://docs.makie.org/stable/) for plotting, together with a small helper that draws a field with row 1 at the top. Missing values are drawn in white.

```@example hmrf
using HiddenMarkovRandomFields, Distributions, Random, CairoMakie

const LABELCOLOURS = ["#CB3C33", "#389826", "#9558B2", "#4063D8"]

# Draws the matrix A in the grid position pos: labels (integers 1:q) if q is given, data otherwise
function showfield!(pos, A; title = "", q = nothing, colorrange = nothing)
    ax = Axis(pos; title, aspect = DataAspect(), yreversed = true)
    hidedecorations!(ax)
    hidespines!(ax)
    M = permutedims(float.(coalesce.(A, NaN)))
    if isnothing(q)
        colorrange = something(colorrange, extrema(filter(!isnan, M)))
        heatmap!(ax, M; colormap = :viridis, colorrange, nan_color = :white)
    else
        colormap = cgrad(LABELCOLOURS[1:q], q; categorical = true)
        heatmap!(ax, M; colormap, colorrange = (0.5, q + 0.5), nan_color = :white)
    end
    return ax
end
nothing # hide
```

## Unconditional simulation

[`simulatehiddenpotts`](@ref) simulates labels `Y` and data `Z` on an `nrow × ncol` grid, given the inverse temperature `β` and one emission distribution per label. Each chain can have its own parameters. Below, one call simulates a field for each of several values of β, given as a vector with one value per chain. The labels (top) and data (bottom) are shown for $q = 3$ labels with Gaussian emissions. The critical value for $q = 3$ is $\beta_c = \log(1 + \sqrt{3}) \approx 1.005$.

```@example hmrf
distributions = [Normal(-1, 0.4), Normal(0, 0.4), Normal(1, 0.4)]
βs = [0.6, 0.8, 1.0, 1.2, 1.4]
Y, Z = simulatehiddenpotts(100, 100, βs, distributions; rng = Xoshiro(1))

fig = Figure(size = (1000, 420))
for (j, β) in enumerate(βs)
    showfield!(fig[1, j], Y[:, :, 1, j]; title = "β = $β", q = 3)
    showfield!(fig[2, j], Z[:, :, 1, j]; colorrange = (-2, 2))
end
fig
```

The emission distributions can also differ between chains, given as a vector with one vector of distributions per chain. This simulates, for example, the training data of a neural estimator in one call, with each field drawn from its own parameter values. Below, 500 fields of 64 × 64 pixels each have their own β, uniform on (0, 1.5), and their own standard deviation σ of the Gaussian emissions, uniform on (0.1, 0.6). Five of them are shown, in increasing order of β.

```@example hmrf
K = 500
β_train = 1.5 .* rand(Xoshiro(2), K)
σ_train = 0.1 .+ 0.5 .* rand(Xoshiro(3), K)
distributions_train = [[Normal(-1, σ), Normal(0, σ), Normal(1, σ)] for σ in σ_train]
Y_train, Z_train = simulatehiddenpotts(64, 64, β_train, distributions_train; num_iterations = 100, rng = Xoshiro(4))

fig = Figure(size = (1000, 230))
for (j, k) in enumerate(sortperm(β_train)[[1, 125, 250, 375, 500]])
    title = "β = $(round(β_train[k]; digits = 2)), σ = $(round(σ_train[k]; digits = 2))"
    showfield!(fig[1, j], Z_train[:, :, 1, k]; title, colorrange = (-2, 2))
end
fig
```

## Conditional simulation

Given incomplete data, in which `missing` marks the values to simulate, [`simulatehiddenpotts`](@ref) simulates the missing values conditionally on the observed ones. Here, a disc is removed from the top-right corner of the image, together with 20% of the remaining values at random. Each of 100 independent chains then gives one conditional simulation.

```@example hmrf
β = 1.0
z = simulatehiddenpotts(100, 100, β, distributions; rng = Xoshiro(10)).Z[:, :, 1, 1]

Z₁ = Matrix{Union{Missing, Float64}}(z)
Z₁[[hypot(i - 1, j - 100) < 45 for i = 1:100, j = 1:100]] .= missing
Z₁[rand(Xoshiro(11), 100, 100) .< 0.2] .= missing

Y, Z = simulatehiddenpotts(Z₁, β, distributions; nchains = 100, num_iterations = 100, rng = Xoshiro(12))

fig = Figure(size = (1000, 240))
showfield!(fig[1, 1], z; title = "Complete data", colorrange = (-2, 2))
showfield!(fig[1, 2], Z₁; title = "Incomplete data", colorrange = (-2, 2))
showfield!(fig[1, 3], Z[:, :, 1, 1]; title = "Conditional simulation 1", colorrange = (-2, 2))
showfield!(fig[1, 4], Z[:, :, 1, 2]; title = "Conditional simulation 2", colorrange = (-2, 2))
showfield!(fig[1, 5], mean(Z; dims = 4)[:, :, 1, 1]; title = "Conditional mean", colorrange = (-2, 2))
fig
```

The observed values are identical in every conditional simulation. Only the missing values vary.

## The Potts model

[`simulatepotts`](@ref) simulates the Potts model itself, in which the labels are observed directly. It uses the same sampler, keyword arguments and output layout as [`simulatehiddenpotts`](@ref). The following figure shows unconditional simulations from the two-state model (the Ising model, $\beta_c \approx 0.881$) for increasing β, similar to Figure 8.8 of Marin and Robert (2007, *Bayesian Core*).

```@example hmrf
fig = Figure(size = (1000, 420))
for (k, β) in enumerate(0.3:0.1:1.2)
    Y = simulatepotts(100, 100, 2, β; rng = Xoshiro(k))
    showfield!(fig[fld1(k, 5), mod1(k, 5)], Y[:, :, 1, 1]; title = "β = $β", q = 2)
end
fig
```

For conditional simulation, the known labels are held fixed and `missing` marks the labels to simulate. The number of labels `q` is always given explicitly, so labels absent from the observed part of the field can still be simulated.

```@example hmrf
Y = simulatepotts(100, 100, 3, 1.0; rng = Xoshiro(23))[:, :, 1, 1]
Y₁ = Matrix{Union{Missing, Int}}(Y)
Y₁[31:70, 31:70] .= missing

Ycond = simulatepotts(Y₁, 3, 1.0; nchains = 3, num_iterations = 100, rng = Xoshiro(21))

fig = Figure(size = (1000, 240))
showfield!(fig[1, 1], Y; title = "Complete labels", q = 3)
showfield!(fig[1, 2], Y₁; title = "Incomplete labels", q = 3)
for c = 1:3
    showfield!(fig[1, 2 + c], Ycond[:, :, 1, c]; title = "Conditional simulation $c", q = 3)
end
fig
```

## Point-mass emissions: a sea-ice example

The emission distributions may include point masses. Consider sea-ice concentration, a proportion in $[0, 1]$, that is exactly 0 over open water and exactly 1 over full ice. One way to model it uses four labels: partial ice with low and high concentrations (Beta emissions), open water (`Dirac(0.0)`) and full ice (`Dirac(1.0)`).

```@example hmrf
seaice = [Beta(2, 5), Beta(5, 2), Dirac(0.0), Dirac(1.0)]
z = simulatehiddenpotts(100, 100, 1.05, seaice; rng = Xoshiro(30)).Z[:, :, 1, 1]
(open_water = count(==(0), z), full_ice = count(==(1), z), partial_ice = count(x -> 0 < x < 1, z))
```

In conditional simulation, an observed 0 or 1 identifies its label exactly, so these pixels are held fixed. They then constrain the labels of the missing pixels around them.

```@example hmrf
Z₁ = Matrix{Union{Missing, Float64}}(z)
Z₁[21:60, 41:90] .= missing

Y, Z = simulatehiddenpotts(Z₁, 1.05, seaice; nchains = 2, num_iterations = 100, rng = Xoshiro(31))

fig = Figure(size = (800, 240))
showfield!(fig[1, 1], z; title = "Complete data", colorrange = (0, 1))
showfield!(fig[1, 2], Z₁; title = "Incomplete data", colorrange = (0, 1))
showfield!(fig[1, 3], Z[:, :, 1, 1]; title = "Conditional simulation 1", colorrange = (0, 1))
showfield!(fig[1, 4], Z[:, :, 1, 2]; title = "Conditional simulation 2", colorrange = (0, 1))
fig
```

## Burn-in and Swendsen–Wang updates

The keyword argument `save_at` returns the state of each chain after selected iterations, where iteration 0 is the initial state. Here it is used to track the proportion of neighbouring pairs with equal labels over 300 iterations, at β = 1.3 (above the critical value). Four chains are run with Gibbs sweeps only (`sw_every = 0`), and four with a Swendsen–Wang sweep in every iteration (`sw_every = 1`, the default). The Gibbs chains keep the separate domains of their initial states. The Swendsen–Wang updates change whole clusters at once, so those chains quickly reach states dominated by a single label.

```@example hmrf
likepairs(Y) = count(Y[1:(end - 1), :] .== Y[2:end, :]) + count(Y[:, 1:(end - 1)] .== Y[:, 2:end])
n = 80
iterations = 0:5:300

fig = Figure(size = (1000, 340))
ax = Axis(fig[1, 1]; xlabel = "Iteration", ylabel = "Proportion of like neighbour pairs")
for (k, (sw_every, name)) in enumerate(((0, "Gibbs"), (1, "Gibbs + Swendsen–Wang")))
    Y = simulatepotts(n, n, 3, 1.3; nchains = 4, sw_every, num_iterations = 300, save_at = iterations, rng = Xoshiro(40))
    for c = 1:4
        trace = [likepairs(Y[:, :, s, c]) / (2n * (n - 1)) for s in eachindex(iterations)]
        lines!(ax, iterations, trace; color = Makie.wong_colors()[k], label = name)
    end
    showfield!(fig[1, k + 1], Y[:, :, end, 1]; title = "$name: iteration 300", q = 3)
end
Legend(fig[2, 1], ax; merge = true, orientation = :horizontal, framevisible = false)
colsize!(fig.layout, 1, Relative(0.45))
fig
```

## Persistent chains

[`simulatehiddenpotts`](@ref) and [`simulatepotts`](@ref) start new chains in every call. Algorithms such as Monte Carlo EM ([Wei and Tanner, 1990](https://doi.org/10.1080/01621459.1990.10474930)), including the neural EM algorithm ([Sainsbury-Dale et al., 2025](https://arxiv.org/abs/2501.04330)), need to continue the same chains while the parameters change. They can use the persistent-chain interface (see [API](@ref)), which separates three kinds of quantity:

- the data layout, [`PottsLattice`](@ref HiddenMarkovRandomFields.PottsLattice), built once per data set;
- the parameter-dependent quantities, [`PottsParams`](@ref HiddenMarkovRandomFields.PottsParams), rebuilt whenever the parameters change; and
- the chains, [`PottsChain`](@ref HiddenMarkovRandomFields.PottsChain), as many as needed.

```@example hmrf
using HiddenMarkovRandomFields: PottsLattice, PottsParams, PottsChain, run!, getlabels!, drawZ!

# Incomplete data
z = simulatehiddenpotts(64, 64, 1.1, distributions; rng = Xoshiro(50)).Z[:, :, 1, 1]
Z₁ = Matrix{Union{Missing, Float64}}(z)
Z₁[rand(Xoshiro(51), 64, 64) .< 0.3] .= missing

# Once per data set: the lattice and the chains
lattice = PottsLattice(Z₁, distributions)
chains = [PottsChain(lattice, Xoshiro(c)) for c = 1:10]

# Burn in under initial parameter values (the first call to run! initialises each chain)
params = PottsParams(lattice, 0.5, distributions)
foreach(chain -> run!(chain, lattice, params, 50), chains)

# New parameter values: the chains continue from their current states
params = PottsParams(lattice, 1.0, [Normal(-1.1, 0.5), Normal(0.1, 0.5), Normal(0.9, 0.5)])
Threads.@threads for chain in chains # chains can be updated in parallel
    run!(chain, lattice, params, 5)
end

# The current labels, and a completion of the data, from each chain
labels = [getlabels!(zeros(Int, size(Z₁)), chain, lattice) for chain in chains]
completed = [drawZ!(coalesce.(Z₁, 0.0), chain, lattice, params) for chain in chains]
size(completed), size(completed[1])
```

## Simulation on a GPU

With a GPU package loaded, the keyword argument `backend` runs all the chains together on the GPU, and the results stay there (see [GPU sampling](@ref)). The code below needs an NVIDIA GPU and CUDA.jl, so it isn't run when these pages are built.

```julia
using HiddenMarkovRandomFields, Distributions, CUDA
using HiddenMarkovRandomFields: PottsLattice, PottsParams, PottsChains, run!, drawZ!

distributions = [Normal(-1, 0.5), Normal(0, 0.5), Normal(1, 0.5)]
Z₁ = simulatehiddenpotts(64, 64, 1.1, distributions).Z[:, :, 1, 1]
Z₁ = Matrix{Union{Missing, Float64}}(Z₁)
Z₁[rand(64, 64) .< 0.3] .= missing

# 1000 completions of the data, simulated on the GPU: Y and Z are CuArrays
Y, Z = simulatehiddenpotts(Z₁, 1.1, distributions; nchains = 1000, backend = CUDABackend())
Z_cpu = Array(Z) # a copy on the CPU, if needed

# Training data: 1000 fields, each with its own β and emission standard deviation
βs, σs = 1.5 .* rand(1000), 0.1 .+ 0.5 .* rand(1000)
Y_train, Z_train = simulatehiddenpotts(64, 64, βs, [[Normal(-1, σ), Normal(0, σ), Normal(1, σ)] for σ in σs]; backend = CUDABackend())

# Persistent chains on the GPU, continued under new parameter values
lattice = PottsLattice(Z₁, distributions)
chains = PottsChains(lattice, 1000, CUDABackend())
run!(chains, lattice, PottsParams(lattice, 0.5, distributions), 50)
params = PottsParams(lattice, 1.0, distributions)
run!(chains, lattice, params, 5)
completed = CuArray(repeat(coalesce.(Z₁, 0.0), 1, 1, 1000)) # observed values, on the GPU
drawZ!(completed, chains, lattice, params)                   # missing values drawn on the GPU
```
