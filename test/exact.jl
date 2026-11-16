# Exact-target tests of the (hidden) Potts sampler.
#
# On grids small enough to enumerate every label configuration, the exact label marginals, the expected number of
# like-neighbour pairs, and the moments of the missing data are compared with Monte Carlo estimates:
#   - long runs of a single persistent chain (Gibbs only, and Gibbs + Swendsen–Wang), allowing for Monte Carlo error
#     via batch means; and
#   - the final states of many independent chains from simulatehiddenpotts and simulatepotts.

using HiddenMarkovRandomFields, Test, Random, Statistics
using HiddenMarkovRandomFields: PottsLattice, PottsParams, PottsChain, initialize!, run!, getlabels!, drawZ!
using Distributions: Normal, Beta, Dirac, logpdf, mean, var

# Exact computations by enumeration: likepairs, exactposterior, and the comparison of independent chains with them
@isdefined(exactposterior) || include("enumeration.jl")

# ---- Persistent chain: batch means ----

# Runs one chain for nbatch × batchlength iterations, returning batch means of the label indicators (nrow × ncol × q × nbatch),
# of likepairs (nbatch), and of Z and Z² at the missing pixels (nmissing × nbatch each)
function samplerbatches(lat, par; sw_every, nbatch = 50, batchlength = 4000, rng = Xoshiro(1), burn = 1000)
    q = par.q
    chain = PottsChain(lat, Xoshiro(rand(rng, UInt64)))
    initialize!(chain, lat, par)
    run!(chain, lat, par, burn; sw_every = sw_every)
    Y = zeros(Int, lat.nrow, lat.ncol)
    Zbuf = zeros(lat.nrow, lat.ncol)
    nmiss = length(lat.missing_lin)
    I = zeros(lat.nrow, lat.ncol, q, nbatch)
    S = zeros(nbatch)
    Z1 = zeros(nmiss, nbatch)
    Z2 = zeros(nmiss, nbatch)
    hasZ = !isnothing(par.samplers)
    for b = 1:nbatch
        for _ = 1:batchlength
            run!(chain, lat, par, 1; sw_every = sw_every)
            getlabels!(Y, chain, lat)
            for i in eachindex(Y)
                I[i + (Y[i] - 1) * length(Y) + (b - 1) * length(Y) * q] += 1
            end
            S[b] += likepairs(Y)
            if hasZ
                drawZ!(Zbuf, chain, lat, par)
                for (t, i) in enumerate(lat.missing_lin)
                    Z1[t, b] += Zbuf[i]
                    Z2[t, b] += Zbuf[i]^2
                end
            end
        end
    end
    return (I = I ./ batchlength, S = S ./ batchlength, Z1 = Z1 ./ batchlength, Z2 = Z2 ./ batchlength)
end

# |batch-mean estimate - exact| < k standard errors (plus a small absolute tolerance for quantities with ~zero variance)
function agrees(batches::AbstractArray, exact; k = 4, atol = 2e-3)
    est = mean(batches; dims = ndims(batches))
    se = std(batches; dims = ndims(batches)) ./ sqrt(size(batches, ndims(batches)))
    est, se = dropdims(est; dims = ndims(batches)), dropdims(se; dims = ndims(batches))
    ok = abs.(est .- exact) .<= k .* se .+ atol
    if !all(ok)
        bad = findall(.!ok)
        @warn "Disagreement" bad est[bad] exact[bad] se[bad]
    end
    return all(ok)
end

# Lattice for the data Z, with the default number of blocks or with `nblocks` blocks
lattice(Z, distributions, nblocks) = isnothing(nblocks) ? PottsLattice(Z, distributions) : PottsLattice(Z, distributions; nblocks = nblocks)

# Hidden Potts model with data Z (missing values, and observed values equal to Dirac atoms, allowed)
function checkmodel(Z, β, distributions; nblocks = nothing, nbatch = 50, batchlength = 4000)
    lat = lattice(Z, distributions, nblocks)
    par = PottsParams(lat, β, distributions)
    fixedlabel = zeros(Int, size(Z))
    for (p, l) in zip(lat.fixed_pad, lat.fixed_label)
        i, j = (p - 1) % lat.stride, (p - 1) ÷ lat.stride # padded (row, col) - 1
        fixedlabel[i, j] = l
    end
    marg, ES = exactposterior(Z, β, distributions, fixedlabel)
    # Moments of the missing data
    μ = [mean(d) for d in distributions]
    m2 = [var(d) + mean(d)^2 for d in distributions]
    n = length(Z)
    EZ = [sum(marg[i + (k - 1) * n] * μ[k] for k in eachindex(distributions)) for i in lat.missing_lin]
    EZ2 = [sum(marg[i + (k - 1) * n] * m2[k] for k in eachindex(distributions)) for i in lat.missing_lin]
    for sw_every in (0, 1)
        r = samplerbatches(lat, par; sw_every = sw_every, nbatch = nbatch, batchlength = batchlength)
        @test agrees(r.I, marg)
        @test agrees(r.S, ES; atol = 5e-3)
        @test agrees(r.Z1, EZ; atol = 5e-3)
        @test agrees(r.Z2, EZ2; atol = 5e-3)
    end
end

# Potts model with known labels in `grid` (`missing` marks the labels to simulate; all missing: unconditional)
function checkpotts(grid, q, β; nblocks = nothing)
    fixedlabel = map(x -> ismissing(x) ? 0 : x, grid)
    lat = PottsLattice(ismissing.(grid), fixedlabel, zeros(size(grid)); nblocks = something(nblocks, 1))
    par = PottsParams(lat, β, q)
    placeholder = [Normal(0.0, 1.0) for _ = 1:q] # no observed pixels contribute a likelihood
    marg, ES = exactposterior(Matrix{Union{Missing, Float64}}(missing, size(grid)...), β, placeholder, fixedlabel)
    for sw_every in (0, 1)
        r = samplerbatches(lat, par; sw_every = sw_every)
        @test agrees(r.I, marg)
        @test agrees(r.S, ES; atol = 5e-3)
    end
end

@testset "exact targets: persistent chain" begin
    rng = Xoshiro(2025)

    @testset "3×3, q = 3, Normal emissions" begin
        distributions = [Normal(-1.0, 0.6), Normal(0.0, 0.6), Normal(1.0, 0.6)]
        Z = Matrix{Union{Missing, Float64}}(randn(rng, 3, 3) .* 0.8)
        Z[1, 1] = missing
        Z[2, 2] = missing
        Z[3, 1] = missing
        checkmodel(Z, 0.8, distributions)
    end

    @testset "3×3, q = 3, strong coupling (β = 1.5)" begin
        distributions = [Normal(-1.0, 0.5), Normal(0.0, 0.5), Normal(1.0, 0.5)]
        Z = Matrix{Union{Missing, Float64}}(randn(rng, 3, 3) .* 0.8)
        Z[1:2, 2:3] .= missing
        checkmodel(Z, 1.5, distributions)
    end

    @testset "3×3, q = 4, Beta + Dirac(0) + Dirac(1) emissions (sea-ice style)" begin
        distributions = [Beta(2.0, 5.0), Beta(5.0, 2.0), Dirac(0.0), Dirac(1.0)]
        Z = Matrix{Union{Missing, Float64}}(rand(rng, 3, 3))
        Z[1, 1] = 0.0      # fixed: Dirac(0)
        Z[3, 3] = 1.0      # fixed: Dirac(1)
        Z[2, 2] = missing
        Z[1, 3] = missing
        Z[3, 2] = missing
        checkmodel(Z, 1.0, distributions)
    end

    @testset "4×4, q = 2" begin
        distributions = [Normal(0.0, 1.0), Normal(1.5, 1.0)]
        Z = Matrix{Union{Missing, Float64}}(randn(rng, 4, 4))
        Z[rand(rng, 4, 4) .< 0.5] .= missing
        checkmodel(Z, 0.7, distributions; batchlength = 3000)
    end

    @testset "1×7 strip and 7×1 strip, q = 3" begin
        distributions = [Normal(-1.0, 0.7), Normal(0.0, 0.7), Normal(1.0, 0.7)]
        z = Vector{Union{Missing, Float64}}(randn(rng, 7))
        z[[2, 5, 6]] .= missing
        checkmodel(reshape(z, 1, 7), 1.2, distributions)
        checkmodel(reshape(z, 7, 1), 1.2, distributions)
    end

    @testset "All pixels missing (unconditional), 3×3, q = 3" begin
        distributions = [Normal(-1.0, 0.5), Normal(0.0, 0.5), Normal(1.0, 0.5)]
        Z = Matrix{Union{Missing, Float64}}(missing, 3, 3)
        checkmodel(Z, 1.0, distributions)
    end

    @testset "Potts model conditional on known labels" begin
        grid = Matrix{Union{Missing, Int}}([1 missing 2; missing missing 3; 1 missing missing])
        checkpotts(grid, 3, 0.9)
    end

    @testset "Potts model, unconditional" begin
        checkpotts(Matrix{Union{Missing, Int}}(missing, 3, 3), 3, 1.1)
    end
end

# One block per column: every right bond crosses a block boundary, so that clusters are joined across blocks (the
# results do not depend on the number of tasks, which is tested in behaviour.jl)
@testset "exact targets: persistent chain, one block per column" begin
    rng = Xoshiro(2026)

    @testset "3×3, q = 3, Normal emissions" begin
        distributions = [Normal(-1.0, 0.6), Normal(0.0, 0.6), Normal(1.0, 0.6)]
        Z = Matrix{Union{Missing, Float64}}(randn(rng, 3, 3) .* 0.8)
        Z[1, 1] = missing
        Z[2, 2] = missing
        Z[3, 1] = missing
        checkmodel(Z, 0.8, distributions; nblocks = 3)
    end

    @testset "3×3, q = 3, strong coupling (β = 1.5)" begin
        distributions = [Normal(-1.0, 0.5), Normal(0.0, 0.5), Normal(1.0, 0.5)]
        Z = Matrix{Union{Missing, Float64}}(randn(rng, 3, 3) .* 0.8)
        Z[1:2, 2:3] .= missing
        checkmodel(Z, 1.5, distributions; nblocks = 3)
    end

    @testset "3×3, q = 4, Beta + Dirac(0) + Dirac(1) emissions" begin
        distributions = [Beta(2.0, 5.0), Beta(5.0, 2.0), Dirac(0.0), Dirac(1.0)]
        Z = Matrix{Union{Missing, Float64}}(rand(rng, 3, 3))
        Z[1, 1] = 0.0      # fixed: Dirac(0)
        Z[2, 3] = 0.0      # fixed: Dirac(0), two blocks to the right
        Z[3, 2] = 1.0      # fixed: Dirac(1)
        Z[2, 2] = missing
        Z[1, 3] = missing
        checkmodel(Z, 1.2, distributions; nblocks = 3)
    end

    @testset "4×4, q = 2" begin
        distributions = [Normal(0.0, 1.0), Normal(1.5, 1.0)]
        Z = Matrix{Union{Missing, Float64}}(randn(rng, 4, 4))
        Z[rand(rng, 4, 4) .< 0.5] .= missing
        checkmodel(Z, 0.9, distributions; nblocks = 4, batchlength = 3000)
    end

    @testset "1×7 strip, q = 3" begin
        distributions = [Normal(-1.0, 0.7), Normal(0.0, 0.7), Normal(1.0, 0.7)]
        z = Vector{Union{Missing, Float64}}(randn(rng, 7))
        z[[2, 5, 6]] .= missing
        checkmodel(reshape(z, 1, 7), 1.2, distributions; nblocks = 7)
    end

    @testset "All pixels missing (unconditional), 3×3, q = 3" begin
        distributions = [Normal(-1.0, 0.5), Normal(0.0, 0.5), Normal(1.0, 0.5)]
        checkmodel(Matrix{Union{Missing, Float64}}(missing, 3, 3), 1.0, distributions; nblocks = 3)
    end

    @testset "Potts model, conditional and unconditional" begin
        checkpotts(Matrix{Union{Missing, Int}}([1 missing 2; missing missing 3; 1 missing missing]), 3, 0.9; nblocks = 3)
        checkpotts(Matrix{Union{Missing, Int}}(missing, 3, 3), 3, 1.1; nblocks = 3)
    end
end

# ---- Independent chains: simulatehiddenpotts and simulatepotts ----

@testset "exact targets: independent chains" begin
    n = 20_000
    q = 3

    @testset "simulatehiddenpotts, conditional" begin
        distributions = [Normal(-1.0, 0.6), Normal(0.0, 0.6), Normal(1.0, 0.6)]
        Z = Matrix{Union{Missing, Float64}}(randn(Xoshiro(11), 3, 3) .* 0.8)
        Z[1, 1] = missing
        Z[2, 2] = missing
        Z[3, 1] = missing
        marg, ES = exactposterior(Z, 0.8, distributions, zeros(Int, 3, 3))
        for sw_every in (0, 1)
            r = simulatehiddenpotts(Z, 0.8, distributions; nchains = n, num_iterations = 50, sw_every = sw_every, rng = Xoshiro(12))
            @test agrees_independent(r.Y, q, marg, ES)
        end
    end

    @testset "simulatepotts, conditional and unconditional" begin
        placeholder = [Normal(0.0, 1.0) for _ = 1:q]
        allmissing = Matrix{Union{Missing, Float64}}(missing, 3, 3)
        grid = Matrix{Union{Missing, Int}}([1 missing 2; missing missing 3; 1 missing missing])
        marg, ES = exactposterior(allmissing, 0.9, placeholder, coalesce.(grid, 0))
        for sw_every in (0, 1)
            Y = simulatepotts(grid, q, 0.9; nchains = n, num_iterations = 50, sw_every = sw_every, rng = Xoshiro(13))
            @test agrees_independent(Y, q, marg, ES)
        end
        marg, ES = exactposterior(allmissing, 1.1, placeholder, zeros(Int, 3, 3))
        for sw_every in (0, 1)
            Y = simulatepotts(3, 3, q, 1.1; nchains = n, num_iterations = 50, sw_every = sw_every, rng = Xoshiro(14))
            @test agrees_independent(Y, q, marg, ES)
        end
    end

    @testset "parameters per chain" begin
        # Two groups of chains, alternating: β = 0.5 and the emission distributions dA, and β = 1.2 and dB
        odd = 1:2:n
        even = 2:2:n
        βs = [isodd(c) ? 0.5 : 1.2 for c = 1:n]
        dA = [Normal(-1.0, 0.5), Normal(0.0, 0.5), Normal(1.0, 0.5)]
        dB = [Normal(4.0, 0.3), Normal(5.0, 0.3), Normal(6.0, 0.3)]
        dists = [isodd(c) ? dA : dB for c = 1:n]
        placeholder = [Normal(0.0, 1.0) for _ = 1:q]
        allmissing = Matrix{Union{Missing, Float64}}(missing, 3, 3)
        nofixed = zeros(Int, 3, 3)
        # Unconditional simulation: the labels follow each chain's Potts model, and its data its emission distributions
        for sw_every in (0, 1)
            r = simulatehiddenpotts(3, 3, βs, dists; num_iterations = 50, sw_every = sw_every, rng = Xoshiro(15))
            @test size(r.Y) == size(r.Z) == (3, 3, 1, n)
            for (group, β, d) in ((odd, 0.5, dA), (even, 1.2, dB))
                marg, ES = exactposterior(allmissing, β, placeholder, nofixed)
                @test agrees_independent(r.Y[:, :, :, group], q, marg, ES)
                @test agrees_missingdata(r.Z[:, :, :, group], allmissing, d, marg)
            end
        end
        # Conditional simulation with β per chain, and with emission distributions per chain
        Z = Matrix{Union{Missing, Float64}}(randn(Xoshiro(16), 3, 3) .* 0.8)
        Z[1, 1] = missing
        Z[2, 3] = missing
        r = simulatehiddenpotts(Z, βs, dA; num_iterations = 50, rng = Xoshiro(17))
        for (group, β) in ((odd, 0.5), (even, 1.2))
            marg, ES = exactposterior(Z, β, dA, nofixed)
            @test agrees_independent(r.Y[:, :, :, group], q, marg, ES)
        end
        dC = [Normal(-0.5, 1.0), Normal(0.5, 0.4), Normal(1.5, 0.8)]
        r = simulatehiddenpotts(Z, 0.8, [isodd(c) ? dA : dC for c = 1:n]; num_iterations = 50, rng = Xoshiro(18))
        for (group, d) in ((odd, dA), (even, dC))
            marg, ES = exactposterior(Z, 0.8, d, nofixed)
            @test agrees_independent(r.Y[:, :, :, group], q, marg, ES)
            @test agrees_missingdata(r.Z[:, :, :, group], Z, d, marg)
        end
        # The Potts model, conditional on known labels, with β per chain
        grid = Matrix{Union{Missing, Int}}([1 missing 2; missing missing 3; 1 missing missing])
        Y = simulatepotts(grid, q, βs; num_iterations = 50, rng = Xoshiro(19))
        for (group, β) in ((odd, 0.5), (even, 1.2))
            marg, ES = exactposterior(allmissing, β, placeholder, coalesce.(grid, 0))
            @test agrees_independent(Y[:, :, :, group], q, marg, ES)
        end
    end
end
