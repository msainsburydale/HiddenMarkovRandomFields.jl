# MCMC for the (hidden) Potts model on a regular nrow × ncol grid with a first-order
# (4-nearest-neighbour) neighbourhood and free boundary:
#
#   p(Y) ∝ exp(β Σ_{i~j} 1{Yᵢ = Yⱼ}),    Zᵢ | Yᵢ = k ~ distributions[k] (independently).
#
# A single core handles conditional and unconditional simulation of the hidden Potts model, and of the Potts model
# itself (known labels are treated as fixed pixels). Each MCMC iteration is an (optional) Swendsen–Wang sweep followed
# by a chequerboard heat-bath Gibbs sweep:
#   - labels are stored as UInt8 in a padded (nrow+2)×(ncol+2) array with a border of zeros,
#     so neighbour look-ups need no bounds checks;
#   - likelihood weights of the observed pixels are computed once per parameter value, and
#     exp(β n) is tabulated for the five possible neighbour counts n = 0, …, 4, so the inner
#     loop contains no exp(), logpdf(), allocation, or dynamic dispatch;
#   - uniforms are generated in bulk per sweep (SIMD Xoshiro);
#   - missing Z values are drawn only for the returned states;
#   - the grid is divided into blocks of consecutive columns, so that a single chain can be updated by several
#     tasks: Gibbs half-sweeps are parallel over blocks, and Swendsen–Wang runs union-find within blocks in
#     parallel and then joins the clusters that cross block boundaries;
#   - each chain has its own Xoshiro streams (one per block, for the sweeps, and one for initialisation and data),
#     seeded from a master RNG, so that results are reproducible and independent of the number of threads;
#   - scratch memory lives in one workspace per task, not in the chains.

"""
	βcrit(q)
Critical inverse temperature `log(1 + √q)` of the `q`-state Potts model on the square lattice. Realisations are
disordered for `β < βcrit(q)` and dominated by large clusters of a single label for `β > βcrit(q)`.
"""
βcrit(q) = log(1 + sqrt(q))

# ---- Lattice: geometry and data layout (independent of the parameters) ----

"""
	PottsLattice(Z::AbstractMatrix, distributions; nblocks)
	PottsLattice(nrow::Integer, ncol::Integer; nblocks)
	PottsLattice(ismiss::AbstractMatrix{Bool}, fixedlabel::AbstractMatrix{<:Integer}, z::AbstractMatrix; nblocks)
Geometry and data layout for MCMC on an `nrow`×`ncol` grid, computed once per data set. Each pixel is
- *missing* (updated, no likelihood contribution);
- *observed* (updated, with likelihood contribution given by column `o+1` of the weight table of [`PottsParams`](@ref)); or
- *fixed* (never updated: an observed value equal to the atom of a `Dirac` emission distribution, or a known label).

The first method is for the hidden Potts model with data `Z` (`missing` marks missing values), the second for
unconditional simulation (every pixel missing), and the third gives full control: `ismiss` marks the missing
pixels, `fixedlabel` the fixed pixels (label > 0) and `z` the observed values of the remaining pixels.

The grid is divided into `nblocks` blocks of consecutive columns, over which the sweeps of a chain are parallelised
(see the keyword argument `ntasks` of [`run!`](@ref)). The default depends only on the number of columns, and is
rarely worth changing. Each block has its own random-number stream, so results depend on `nblocks`, but not on the
number of tasks.
"""
struct PottsLattice
    nrow::Int
    ncol::Int
    stride::Int                       # nrow + 2: column length of the padded label array
    nblocks::Int                      # number of blocks of consecutive columns
    blockcols::Vector{UnitRange{Int}} # columns (unpadded) of each block
    blockupd::NTuple{2, Vector{Int}}  # block b holds update[c][blockupd[c][b]:(blockupd[c][b + 1] - 1)]
    update::NTuple{2, Vector{Int32}}  # padded linear indices of the pixels to update, per chequerboard colour
    wcol::NTuple{2, Vector{Int32}}    # weight-table column of each pixel in `update` (1 = missing pixel)
    col_pad::Vector{Int32}            # weight-table column of every padded index (0 = border or fixed pixel)
    fixlab_pad::Vector{UInt8}         # label of every padded index that is fixed (0 otherwise)
    obs_lin::Vector{Int}              # linear indices (unpadded) of the observed, non-fixed pixels
    zobs::Vector{Float64}             # observed values at obs_lin
    missing_lin::Vector{Int}          # linear indices (unpadded) of the missing pixels
    missing_pad::Vector{Int32}        # padded linear indices of the missing pixels
    fixed_pad::Vector{Int32}          # padded linear indices of the fixed pixels
    fixed_label::Vector{UInt8}        # labels of the fixed pixels
end

# Pixel (i, j) of the unpadded grid ↦ linear index in the padded array
@inline _padindex(i, j, stride) = j * stride + i + 1

# Default number of blocks: about 16 columns per block, and at most 32 blocks (more blocks make the sequential merge
# of a Swendsen–Wang sweep more expensive, which outweighed the gain from finer parallelism with 32 threads)
_default_nblocks(ncol::Integer) = clamp(ncol ÷ 16, 1, 32)

function PottsLattice(ismiss::AbstractMatrix{Bool}, fixedlabel::AbstractMatrix{<:Integer}, z::AbstractMatrix; nblocks::Integer = _default_nblocks(size(ismiss, 2)))
    nrow, ncol = size(ismiss)
    size(fixedlabel) == size(z) == (nrow, ncol) || throw(DimensionMismatch("ismiss, fixedlabel and z must have the same size; got $(size(ismiss)), $(size(fixedlabel)) and $(size(z))"))
    all(l -> 0 <= l <= 255, fixedlabel) || throw(ArgumentError("fixed labels must be between 1 and 255 (0 = not fixed)"))
    1 <= nblocks <= max(ncol, 1) || throw(ArgumentError("nblocks must be between 1 and the number of columns ($ncol); got $nblocks"))
    blockcols = [(div((b - 1) * ncol, nblocks) + 1):div(b * ncol, nblocks) for b = 1:nblocks]
    blockupd = ([1], [1])
    stride = nrow + 2
    npad = stride * (ncol + 2)
    update = (Int32[], Int32[])
    wcol = (Int32[], Int32[])
    col_pad = zeros(Int32, npad)
    fixlab_pad = zeros(UInt8, npad)
    obs_lin = Int[]
    zobs = Float64[]
    missing_lin = Int[]
    missing_pad = Int32[]
    fixed_pad = Int32[]
    fixed_label = UInt8[]
    L = LinearIndices((nrow, ncol))
    for cols in blockcols # the pixels of each block are consecutive in update[c]
        for j in cols, i = 1:nrow
            p = _padindex(i, j, stride)
            c = iseven(i + j) ? 1 : 2
            if ismiss[i, j]
                push!(missing_lin, L[i, j])
                push!(missing_pad, p)
                push!(update[c], p)
                push!(wcol[c], 1)
                col_pad[p] = 1
            elseif fixedlabel[i, j] > 0
                push!(fixed_pad, p)
                push!(fixed_label, fixedlabel[i, j])
                fixlab_pad[p] = fixedlabel[i, j]
            else
                push!(obs_lin, L[i, j])
                push!(zobs, z[i, j])
                o = length(obs_lin) + 1
                push!(update[c], p)
                push!(wcol[c], o)
                col_pad[p] = o
            end
        end
        push!(blockupd[1], length(update[1]) + 1)
        push!(blockupd[2], length(update[2]) + 1)
    end
    return PottsLattice(nrow, ncol, stride, Int(nblocks), blockcols, blockupd, update, wcol, col_pad, fixlab_pad, obs_lin, zobs, missing_lin, missing_pad, fixed_pad, fixed_label)
end

# Hidden Potts data: missing entries are updated; observed values equal to the atom of a Dirac emission distribution are fixed
function PottsLattice(Z::AbstractMatrix, distributions; kwargs...)
    ismiss = ismissing.(Z)
    atoms = _atoms(distributions)
    fixedlabel = zeros(UInt8, size(Z))
    if !isempty(atoms)
        for I in eachindex(Z)
            z = Z[I]
            ismissing(z) && continue
            for (k, v) in atoms
                if z == v
                    fixedlabel[I] = k
                    break
                end
            end
        end
    end
    z = map(x -> ismissing(x) ? 0.0 : Float64(x), Z)
    return PottsLattice(ismiss, fixedlabel, z; kwargs...)
end

# All pixels missing (marginal simulation)
PottsLattice(nrow::Integer, ncol::Integer; kwargs...) = PottsLattice(trues(nrow, ncol), zeros(UInt8, nrow, ncol), zeros(nrow, ncol); kwargs...)

# ---- Parameters: quantities that depend on β and the emission distributions ----

"""
	PottsParams(lattice, β, distributions)
	PottsParams(lattice, β, q::Integer)
Parameter-dependent quantities for [`PottsLattice`](@ref), computed once per parameter value: the number of labels `q`
(always `length(distributions)`, never inferred from the data), the table `E[n+1] = exp(βn)` for the possible
neighbour counts `n = 0,…,4`, the Swendsen–Wang bond probability `1 - exp(-β)`, and the likelihood weights `W`
(`q`×(`nobs`+1)): column 1 ≡ 1 (missing pixels) and `W[k, o+1] = exp(ℓₖ - maxₖ ℓₖ)` with `ℓₖ` the log-density of the
`o`th observed value under `distributions[k]`. The second method is for the Potts model itself (no emission
distributions).
"""
struct PottsParams{S, D}
    q::Int
    β::Float64
    E::NTuple{5, Float64}
    pbond::Float64
    W::Matrix{Float64}
    logW::Matrix{Float64}
    samplers::S
    distributions::D # the emission distributions (nothing for the Potts model), from which data are drawn on a GPU
end

_exptable(β) = ntuple(n -> exp(β * (n - 1)), 5)

function _checkq(q, lat)
    1 <= q <= 255 || throw(ArgumentError("the number of labels must be between 1 and 255; got q = $q"))
    isempty(lat.fixed_label) || maximum(lat.fixed_label) <= q || throw(ArgumentError("a fixed label exceeds the number of labels q = $q"))
end

function PottsParams(lat::PottsLattice, β::Real, distributions::AbstractVector)
    q = length(distributions)
    _checkq(q, lat)
    isfinite(β) && β >= 0 || throw(ArgumentError("β must be finite and non-negative; got β = $β"))
    nobs = length(lat.obs_lin)
    logW = zeros(Float64, q, nobs + 1)
    for k = 1:q
        _loglik!(logW, k, distributions[k], lat.zobs) # function barrier: type-stable loop per distribution
    end
    W = ones(Float64, q, nobs + 1)
    @inbounds for o = 2:(nobs + 1)
        m = -Inf
        for k = 1:q
            m = max(m, logW[k, o])
        end
        if isnan(m) || m == -Inf
            throw(ArgumentError("the observed value $(lat.zobs[o - 1]) has zero (or undefined) likelihood under every emission distribution"))
        elseif m == Inf # a density that is infinite at the observed value takes all the mass
            for k = 1:q
                inf = logW[k, o] == Inf
                logW[k, o] = inf ? 0.0 : -Inf
                W[k, o] = inf ? 1.0 : 0.0
            end
        else
            for k = 1:q
                ℓ = logW[k, o] - m
                logW[k, o] = ℓ
                W[k, o] = exp(ℓ)
            end
        end
    end
    β = Float64(β)
    return PottsParams(q, β, _exptable(β), -expm1(-β), W, logW, map(sampler, distributions), distributions)
end

function _loglik!(logW, k, d, zobs)
    @inbounds for o in eachindex(zobs)
        logW[k, o + 1] = logpdf(d, zobs[o])
    end
    return logW
end

function PottsParams(lat::PottsLattice, β::Real, q::Integer)
    _checkq(q, lat)
    isfinite(β) && β >= 0 || throw(ArgumentError("β must be finite and non-negative; got β = $β"))
    isempty(lat.obs_lin) || throw(ArgumentError("emission distributions are needed when there are observed (non-fixed) pixels"))
    β = Float64(β)
    return PottsParams(Int(q), β, _exptable(β), -expm1(-β), ones(Float64, q, 1), zeros(Float64, q, 1), nothing, nothing)
end

# The parameters `par` with the inverse temperature β instead: the same likelihood weights, samplers and distributions
function _withβ(par::PottsParams, β::Real)
    isfinite(β) && β >= 0 || throw(ArgumentError("β must be finite and non-negative; got β = $β"))
    β = Float64(β)
    return PottsParams(par.q, β, _exptable(β), -expm1(-β), par.W, par.logW, par.samplers, par.distributions)
end

# ---- Parameters per chain ----

# The parameters of a set of chains: one PottsParams shared by all of them, or a vector with one per chain
const _ChainParams = Union{PottsParams, AbstractVector{<:PottsParams}}
_param(par::PottsParams, _) = par
_param(pars::AbstractVector{<:PottsParams}, c) = pars[c]
_q(par::PottsParams) = par.q
_q(pars::AbstractVector{<:PottsParams}) = first(pars).q

# Whether β has one value per chain (a container with one value is a scalar, as for _scalar), and whether the emission
# distributions are one vector per chain
_perchainβ(β) = !(β isa Number) && length(β) > 1
_perchain(distributions) = distributions isa AbstractVector && !isempty(distributions) && first(distributions) isa AbstractVector

# Default number of chains: the number of per-chain values of β or of the emission distributions, or 1
function _nchains(β, distributions = nothing)
    nβ = _perchainβ(β) ? length(β) : 1
    nd = !isnothing(distributions) && _perchain(distributions) ? length(distributions) : 1
    nβ > 1 && nd > 1 && nβ != nd && throw(ArgumentError("β has $nβ values, but there are $nd vectors of emission distributions: give one of each per chain"))
    return max(nβ, nd)
end

function _checkperchain(β, nchains::Integer, distributions = nothing)
    nchains > 0 || throw(ArgumentError("nchains must be positive; got $nchains"))
    _perchainβ(β) && length(β) != nchains && throw(ArgumentError("β has $(length(β)) values, but nchains = $nchains: give one value per chain, or one for all chains"))
    if !isnothing(distributions) && _perchain(distributions)
        length(distributions) == nchains || throw(ArgumentError("there are $(length(distributions)) vectors of emission distributions, but nchains = $nchains: give one per chain, or one for all chains"))
        q = length(first(distributions))
        all(d -> length(d) == q, distributions) || throw(ArgumentError("every chain must have the same number of emission distributions (labels)"))
    end
    return nothing
end

# Parameters of `nchains` chains of the hidden Potts model: a single PottsParams if neither β nor the emission
# distributions vary between chains, and otherwise one per chain, which share one table of likelihood weights when only
# β varies
function _chainparams(lat::PottsLattice, β, distributions, nchains::Integer)
    _checkperchain(β, nchains, distributions)
    if _perchain(distributions)
        return [PottsParams(lat, _perchainβ(β) ? β[c] : _scalar(β), distributions[c]) for c = 1:nchains]
    end
    _perchainβ(β) || return PottsParams(lat, _scalar(β), distributions)
    par = PottsParams(lat, first(β), distributions)
    return [_withβ(par, b) for b in β]
end

# The same for the Potts model (q labels, no emission distributions)
function _chainparams(lat::PottsLattice, β, q::Integer, nchains::Integer)
    _checkperchain(β, nchains)
    _perchainβ(β) || return PottsParams(lat, _scalar(β), q)
    par = PottsParams(lat, first(β), q)
    return [_withβ(par, b) for b in β]
end

# The atoms of the Dirac emission distributions, which fix the observed values equal to them: in conditional
# simulation, they must be the same for every chain, since the chains share the lattice
_atoms(distributions) = [(k, d.value) for (k, d) in enumerate(distributions) if d isa Dirac]
function _latticedistributions(distributions)
    _perchain(distributions) || return distributions
    atoms = _atoms(first(distributions))
    all(d -> _atoms(d) == atoms, distributions) || throw(ArgumentError("the Dirac emission distributions must be the same for every chain in conditional simulation, since they fix the observed values equal to their atoms"))
    return first(distributions)
end

# ---- Chains and workspaces ----

"""
	PottsChain(lattice, rng::Xoshiro)
State of one MCMC chain on `lattice`: padded `UInt8` labels, its own random-number generators (`rng`, used for
initialisation and for drawing data, and one per block of the lattice for the sweeps, seeded from `rng`), and the
number of iterations run. A new chain is initialised by [`initialize!`](@ref), or by the first call to [`run!`](@ref).
"""
mutable struct PottsChain
    Y::Matrix{UInt8}
    rng::Xoshiro
    blockrngs::Vector{Xoshiro}
    iter::Int
    initialized::Bool
end
function PottsChain(lat::PottsLattice, rng::Xoshiro)
    blockrngs = [Xoshiro(rand(rng, UInt64)) for _ = 1:lat.nblocks]
    return PottsChain(zeros(UInt8, lat.stride, lat.ncol + 2), rng, blockrngs, 0, false)
end

# Scratch memory, one per task (never part of the chain state)
struct PottsWorkspace
    u::Vector{Float64}       # uniforms
    parent::Vector{Int32}    # union-find forest (Swendsen–Wang)
    clw::Vector{Float64}     # per-cluster log-weights, q per root (Swendsen–Wang)
    forced::Vector{UInt8}    # per-cluster forced label (fixed members)
    newlab::Vector{UInt8}    # per-cluster new label
    bonded::Vector{UInt8}    # whether a pixel has an open bond (Swendsen–Wang)
    merged::Vector{Int32}    # block-local roots merged across block boundaries (Swendsen–Wang)
    mark::Vector{UInt8}      # queued flags (initialisation)
    queue::Vector{Int32}     # FIFO queue (initialisation)
    counts::Vector{Int}      # neighbour counts (initialisation)
end
PottsWorkspace() = PottsWorkspace(Float64[], Int32[], Float64[], UInt8[], UInt8[], UInt8[], Int32[], UInt8[], Int32[], Int[])

# The workspace of the current task (created on first use), for the public chain API
_workspace() = get!(PottsWorkspace, task_local_storage(), :HiddenMarkovRandomFields_PottsWorkspace)::PottsWorkspace

# Consistency of the arguments of the public chain API (the sweeps index with @inbounds)
function _checkchain(chain::PottsChain, lat::PottsLattice)
    size(chain.Y) == (lat.stride, lat.ncol + 2) || throw(DimensionMismatch("the chain was not created for a $(lat.nrow)×$(lat.ncol) lattice"))
    length(chain.blockrngs) == lat.nblocks || throw(DimensionMismatch("the chain was not created for a lattice with $(lat.nblocks) blocks"))
    return nothing
end

function _checkchain(chain::PottsChain, lat::PottsLattice, par::PottsParams)
    _checkchain(chain, lat)
    size(par.W, 2) == length(lat.obs_lin) + 1 || throw(DimensionMismatch("the parameters were not computed for this lattice"))
    return nothing
end

# Applies kernel(b, args...) to the blocks b = 1, …, nblocks in turn
@inline function _foreachblock(kernel::K, nblocks::Int, args::Vararg{Any, N}) where {K, N}
    for b = 1:nblocks
        kernel(b, args...)
    end
    return nothing
end

# Applies kernel(b, args...) to the blocks b = 1, …, nblocks handed out by the shared counter `ticket`: each of the
# tasks that run a phase of a sweep together calls this, and the counter is reset at the barrier that ends the phase
@inline function _phase!(ticket::Threads.Atomic{Int}, nblocks::Int, kernel::K, args::Vararg{Any, N}) where {K, N}
    while (b = Threads.atomic_add!(ticket, 1) + 1) <= nblocks
        kernel(b, args...)
    end
    return nothing
end

# ---- Gibbs (heat-bath) sweep ----

"""
	gibbs_sweep!(chain, lattice, params, workspace, Val(q))
One chequerboard heat-bath Gibbs sweep: for each colour, every updated pixel draws its label from
`P(Yᵢ = k | ⋅) ∝ W[k, ⋅] exp(β nₖ)`, where `nₖ` is the number of neighbours with label `k`. The pixels of one colour
are conditionally independent, so each block of the lattice is updated separately, with uniforms from the block's own
random-number generator (the blocks are updated in parallel by `_run_parallel!`).
"""
function gibbs_sweep!(chain::PottsChain, lat::PottsLattice, par::PottsParams, ws::PottsWorkspace, ::Val{Q}) where {Q}
    for c = 1:2
        upd = lat.update[c]
        isempty(upd) && continue
        u = resize!(ws.u, length(upd))
        _foreachblock(_gibbs_block!, lat.nblocks, chain.Y, chain.blockrngs, lat.blockupd[c], upd, lat.wcol[c], par.W, par.E, u, lat.stride, Val(Q))
    end
    return chain
end

function _gibbs_block!(b, Y, blockrngs, offsets, upd, wcol, W, E, u, stride, ::Val{Q}) where {Q}
    range = @inbounds offsets[b]:(offsets[b + 1] - 1)
    isempty(range) && return Y
    rand!(@inbounds(blockrngs[b]), view(u, range))
    return _gibbs_colour!(Y, upd, wcol, W, E, u, range, stride, Val(Q))
end

function _gibbs_colour!(Y::Matrix{UInt8}, upd::Vector{Int32}, wcol::Vector{Int32}, W::Matrix{Float64}, E::NTuple{5, Float64}, u::Vector{Float64}, range::UnitRange{Int}, stride::Int, ::Val{Q}) where {Q}
    @inbounds for t in range
        p = Int(upd[t])
        j = Int(wcol[t])
        a = Y[p - 1]
        b = Y[p + 1]
        l = Y[p - stride]
        r = Y[p + stride]
        w = ntuple(Val(Q)) do k
            kk = k % UInt8
            n = (a == kk) + (b == kk) + (l == kk) + (r == kk)
            @inbounds W[k, j] * E[n + 1]
        end
        cw = cumsum(w)
        s = u[t] * cw[Q]
        k = 1
        for m = 1:(Q - 1)
            k += s >= cw[m]
        end
        Y[p] = k % UInt8
    end
    return Y
end

# ---- Swendsen–Wang sweep ----

@inline function _find!(parent::Vector{Int32}, x::Integer)
    x = Int32(x)
    @inbounds while parent[x] != x
        g = parent[parent[x]]
        parent[x] = g # path halving
        x = g
    end
    return Int(x)
end

@inline function _maxclw(clw::Vector{Float64}, base::Int, ::Val{Q}) where {Q}
    m = -Inf
    @inbounds for k = 1:Q
        m = max(m, clw[base + k])
    end
    return m
end

# Label of the cluster with root `root`, with probability ∝ exp(clw), from the uniform v
@inline function _draw_cluster_label(clw::Vector{Float64}, root::Int, v::Float64, ::Val{Q}) where {Q}
    base = (root - 1) * Q
    m = _maxclw(clw, base, Val(Q)) # assigned once, so that the closure below does not box it
    w = ntuple(k -> (@inbounds exp(clw[base + k] - m)), Val(Q))
    cw = cumsum(w)
    s = v * cw[Q]
    k = 1
    for mm = 1:(Q - 1)
        k += s >= cw[mm]
    end
    return k % UInt8
end

# Label with probability ∝ W[⋅, col], from the uniform v
@inline function _draw_weighted(W::Matrix{Float64}, col::Int, v::Float64, ::Val{Q}) where {Q}
    w = ntuple(k -> (@inbounds W[k, col]), Val(Q))
    cw = cumsum(w)
    s = v * cw[Q]
    k = 1
    for m = 1:(Q - 1)
        k += s >= cw[m]
    end
    return k % UInt8
end

"""
	sw_sweep!(chain, lattice, params, workspace, Val(q))
One Swendsen–Wang sweep for the hidden Potts model (the Edwards–Sokal construction with an external field):
bonds between equal-label neighbours are opened independently with probability `1 - exp(-β)`, and each resulting
cluster `C` draws a new label with probability `∝ ∏_{i ∈ C} W[k, ⋅]` (missing pixels contribute one). A cluster
containing a fixed pixel keeps that pixel's label. Singleton clusters (pixels with no open bond) are handled
directly, without union-find bookkeeping.

The clusters are found in three phases, so that the blocks of the lattice can be processed in parallel (by
`_run_parallel!`) with results that do not depend on the number of tasks:
1. within each block: bonds are opened and joined by union-find, and the log-weights and forced labels of each
   cluster are accumulated at its block-local root;
2. sequentially: the bonds that cross block boundaries are opened and joined, and the accumulations of the
   block-local roots that are merged are added to the roots of their clusters;
3. within each block: each cluster with its root in the block, and each singleton, draws its new label.
The root of every cluster is its minimum padded index, and every uniform comes from the random-number generator of a
block (two per pixel for the bonds, and one per pixel for the new labels).
"""
function sw_sweep!(chain::PottsChain, lat::PottsLattice, par::PottsParams, ws::PottsWorkspace, ::Val{Q}) where {Q}
    Y = chain.Y
    parent, bonded, clw, forced, newlab, u = _sw_scratch!(ws, lat, length(Y), Val(Q))
    _foreachblock(_sw_block!, lat.nblocks, Y, parent, bonded, clw, forced, u, chain.blockrngs, lat, par.pbond, par.logW, Val(Q))
    _sw_merge!(Y, parent, bonded, clw, forced, ws.merged, u, lat, par.pbond, par.logW, Val(Q))
    _foreachblock(_sw_rootlabels!, lat.nblocks, parent, bonded, clw, forced, newlab, u, lat, Val(Q))
    _foreachblock(_sw_assign!, lat.nblocks, Y, parent, bonded, newlab, u, lat, par.W, Val(Q))
    return chain
end

# Scratch arrays of a Swendsen–Wang sweep, sized for the lattice. For pixel t (column-major order), u[2t - 1] and
# u[2t] are the uniforms of its down and right bonds, and u[2N + t] the uniform of its new label (N pixels).
function _sw_scratch!(ws::PottsWorkspace, lat::PottsLattice, npad::Int, ::Val{Q}) where {Q}
    N = lat.nrow * lat.ncol
    return (resize!(ws.parent, npad), resize!(ws.bonded, npad), resize!(ws.clw, Q * npad), resize!(ws.forced, npad), resize!(ws.newlab, npad), resize!(ws.u, 3N))
end

# A pixel becomes the root of a cluster: no accumulated log-weights, and no forced label
@inline function _initroot!(clw::Vector{Float64}, forced::Vector{UInt8}, p::Integer, ::Val{Q}) where {Q}
    @inbounds for k = 1:Q
        clw[(p - 1) * Q + k] = 0.0
    end
    @inbounds forced[p] = 0x00
    return nothing
end

# Adds the contribution of pixel p (its log-likelihood weights, or its label if it is fixed) to the cluster with root r
@inline function _accumulate!(clw::Vector{Float64}, forced::Vector{UInt8}, r::Integer, p::Integer, lat::PottsLattice, logW::Matrix{Float64}, ::Val{Q}) where {Q}
    col = @inbounds lat.col_pad[p]
    if col >= 2
        @inbounds for k = 1:Q
            clw[(r - 1) * Q + k] += logW[k, col]
        end
    elseif col == 0 # fixed pixel
        @inbounds forced[r] = lat.fixlab_pad[p]
    end
    return nothing
end

# Phase 1, block b: uniforms, bonds within the block, union-find, and accumulation at the block-local roots
function _sw_block!(b, Y, parent, bonded, clw, forced, u, blockrngs, lat, pb, logW, ::Val{Q}) where {Q}
    cols = @inbounds lat.blockcols[b]
    isempty(cols) && return nothing
    nrow, stride = lat.nrow, lat.stride
    N = nrow * lat.ncol
    j1, j2 = first(cols), last(cols)
    rng = @inbounds blockrngs[b]
    rand!(rng, view(u, (2 * (j1 - 1) * nrow + 1):(2 * j2 * nrow)))
    rand!(rng, view(u, (2N + (j1 - 1) * nrow + 1):(2N + j2 * nrow)))
    # Bonds to the neighbours above and to the left, which precede each pixel in column-major order: a pixel joins the
    # cluster of a bonded neighbour, and union-find is needed only when the pixel is bonded to both neighbours and they
    # are in different trees (union by minimum index, so that the root of every cluster is its minimum index). Each
    # edge uses the uniform that belongs to it: the down-bond uniform of the upper neighbour, and the right-bond
    # uniform of the left neighbour. The left neighbours of the block's first column are in the previous block, and
    # those bonds are opened in phase 2; the border labels are zero, so no bond crosses the boundary of the grid. The
    # selections are branch-free where possible, since bonds are too random for branch prediction.
    @inbounds for j in cols
        hasleft = j > j1
        for i = 1:nrow
            p = _padindex(i, j, stride)
            y = Y[p]
            t = 2 * ((j - 1) * nrow + i)
            up = (Y[p - 1] == y) & (u[ifelse(i > 1, t - 3, t - 1)] < pb) # i = 1: the border is above (unused uniform)
            pu = parent[p - 1]
            if hasleft
                left = (Y[p - stride] == y) & (u[t - 2nrow] < pb)
                pl = parent[p - stride]
                bonded[p - stride] |= left % UInt8
            else
                left = false
                pl = Int32(p)
            end
            parent[p] = ifelse(up, pu, ifelse(left, pl, Int32(p)))
            bonded[p] = (up | left) % UInt8
            bonded[p - 1] |= up % UInt8
            if up & left & (pu != pl)
                ra = _find!(parent, p - 1)
                rb = _find!(parent, p - stride)
                if ra != rb
                    lo, hi = minmax(ra, rb)
                    parent[hi] = lo
                    parent[p] = lo
                end
            end
        end
    end

    # Log-weights and forced labels of the block-local clusters with two or more pixels (every member of such a
    # cluster has an open bond). Roots are minimum indices, so a root is visited before the other members of its
    # cluster. Every bonded pixel then points directly to its block-local root.
    @inbounds for j in cols, i = 1:nrow
        p = _padindex(i, j, stride)
        bonded[p] == 0x00 && continue
        r = _find!(parent, p)
        parent[p] = r
        r == p && _initroot!(clw, forced, p, Val(Q))
        _accumulate!(clw, forced, r, p, lat, logW, Val(Q))
    end
    return nothing
end

# Phase 2 (sequential): the bonds that cross block boundaries, which join block-local clusters. A pixel without
# bonds so far first becomes a root; each root that is merged into another cluster is recorded, and its accumulation
# is then added to the root of its final cluster (in a fixed order, so that the sums do not depend on the number of
# tasks). Afterwards, every bonded pixel is at most two steps from its root.
function _sw_merge!(Y, parent, bonded, clw, forced, merged, u, lat, pb, logW, ::Val{Q}) where {Q}
    empty!(merged)
    nrow, stride = lat.nrow, lat.stride
    @inbounds for b = 1:(lat.nblocks - 1)
        j = last(lat.blockcols[b]) # its right neighbour, column j + 1, is in block b + 1
        for i = 1:nrow
            p = _padindex(i, j, stride)
            t = 2 * ((j - 1) * nrow + i)
            ((Y[p + stride] == Y[p]) & (u[t] < pb)) || continue
            for x in (p, p + stride)
                if bonded[x] == 0x00
                    bonded[x] = 0x01
                    _initroot!(clw, forced, x, Val(Q))
                    _accumulate!(clw, forced, x, x, lat, logW, Val(Q))
                end
            end
            ra = _find!(parent, p)
            rb = _find!(parent, p + stride)
            ra == rb && continue
            lo, hi = minmax(ra, rb)
            parent[hi] = lo
            push!(merged, hi)
        end
    end
    @inbounds for m in merged
        g = _find!(parent, m)
        for k = 1:Q
            clw[(g - 1) * Q + k] += clw[(m - 1) * Q + k]
        end
        forced[m] != 0x00 && (forced[g] = forced[m])
        parent[m] = g
    end
    return nothing
end

# Phase 3, block b: new labels of the clusters whose roots are in the block (the forced label, or a draw with
# probability ∝ exp(clw) using the root's label uniform)
function _sw_rootlabels!(b, parent, bonded, clw, forced, newlab, u, lat, ::Val{Q}) where {Q}
    nrow, stride = lat.nrow, lat.stride
    N2 = 2 * nrow * lat.ncol
    @inbounds for j in lat.blockcols[b], i = 1:nrow
        p = _padindex(i, j, stride)
        (bonded[p] != 0x00 && parent[p] == p) || continue
        newlab[p] = forced[p] != 0x00 ? forced[p] : _draw_cluster_label(clw, p, u[N2 + (j - 1) * nrow + i], Val(Q))
    end
    return nothing
end

# Phase 3, block b: every bonded pixel takes the new label of its cluster; singletons draw from their own likelihood
# weights (uniformly if missing), and fixed singletons are unchanged
function _sw_assign!(b, Y, parent, bonded, newlab, u, lat, W, ::Val{Q}) where {Q}
    nrow, stride = lat.nrow, lat.stride
    N2 = 2 * nrow * lat.ncol
    @inbounds for j in lat.blockcols[b], i = 1:nrow
        p = _padindex(i, j, stride)
        if bonded[p] == 0x00
            col = Int(lat.col_pad[p])
            col == 0 && continue
            Y[p] = _draw_weighted(W, col, u[N2 + (j - 1) * nrow + i], Val(Q))
        else
            Y[p] = newlab[parent[parent[p]]]
        end
    end
    return nothing
end

# ---- Initialisation ----

@inline function _argmax_col(W::Matrix{Float64}, col::Integer, q::Integer)
    k̂ = 1
    @inbounds for k = 2:q
        if W[k, col] > W[k̂, col]
            k̂ = k
        end
    end
    return k̂ % UInt8
end

"""
	initialize!(chain, lattice, params)
	initialize!(chains::PottsChains, lattice, params)
Cold start of `chain`, which resets its iteration counter: fixed pixels take their labels, observed pixels take
their maximum-likelihood labels, and missing pixels are filled outward from the assigned pixels (after seeding 1% of
them at random) by a breadth-first flood fill: at high temperature (β < [`βcrit`](@ref)`(q)`) each pixel is drawn
with probability ∝ exp(β nₖ) given its assigned neighbours; at low temperature it takes the neighbour mode (random
tie-break). With no observed or fixed pixels, the labels are uniformly random at high temperature, and random blocks
with 5% random perturbations at low temperature.

Chains on a device ([`PottsChains`](@ref)) are initialised in the same way on the CPU, in parallel over threads, and
then copied to the device.
"""
function initialize!(chain::PottsChain, lat::PottsLattice, par::PottsParams)
    _checkchain(chain, lat, par)
    return _initialize!(chain, lat, par, _workspace())
end

function _initialize!(chain::PottsChain, lat::PottsLattice, par::PottsParams, ws::PottsWorkspace)
    Y = chain.Y
    fill!(Y, 0x00)
    q = par.q
    @inbounds for t in eachindex(lat.fixed_pad)
        Y[lat.fixed_pad[t]] = lat.fixed_label[t]
    end
    @inbounds for c = 1:2
        upd = lat.update[c]
        wc = lat.wcol[c]
        for t in eachindex(upd)
            col = wc[t]
            col == 1 && continue
            Y[upd[t]] = _argmax_col(par.W, col, q)
        end
    end
    _init_missing!(Y, lat, q, par.β, chain.rng, ws)
    chain.iter = 0
    chain.initialized = true
    return chain
end

function _init_missing!(Y::Matrix{UInt8}, lat::PottsLattice, q::Integer, β::Real, rng::AbstractRNG, ws::PottsWorkspace)
    nmiss = length(lat.missing_pad)
    nmiss == 0 && return Y
    if isempty(lat.obs_lin) && isempty(lat.fixed_pad)
        grid = initializepotts(lat.nrow, lat.ncol, q, β; rng = rng)
        @inbounds for j = 1:lat.ncol, i = 1:lat.nrow
            Y[_padindex(i, j, lat.stride)] = grid[i, j]
        end
        return Y
    end

    stride = lat.stride
    # Seed a small fraction of the missing pixels at random
    nseed = ceil(Int, 0.01 * nmiss)
    for _ = 1:nseed
        Y[lat.missing_pad[rand(rng, 1:nmiss)]] = rand(rng, 1:q)
    end

    # Breadth-first flood fill from the assigned pixels
    mark = resize!(ws.mark, length(Y))
    fill!(mark, 0x00)
    queue = empty!(ws.queue)
    counts = resize!(ws.counts, q)
    @inbounds for p in lat.missing_pad
        Y[p] != 0x00 && continue
        if Y[p - 1] != 0x00 || Y[p + 1] != 0x00 || Y[p - stride] != 0x00 || Y[p + stride] != 0x00
            push!(queue, p)
            mark[p] = 0x01
        end
    end
    lowtemp = β >= βcrit(q)
    head = 1
    @inbounds while head <= length(queue)
        p = Int(queue[head])
        head += 1
        fill!(counts, 0)
        for nb in (p - 1, p + 1, p - stride, p + stride)
            l = Y[nb]
            l != 0x00 && (counts[l] += 1)
        end
        Y[p] = (lowtemp ? _mode_label(counts, rng) : _boltzmann_label(counts, β, rng)) % UInt8
        for nb in (p - 1, p + 1, p - stride, p + stride)
            if lat.col_pad[nb] == 1 && Y[nb] == 0x00 && mark[nb] == 0x00
                push!(queue, nb)
                mark[nb] = 0x01
            end
        end
    end

    # Missing pixels that cannot be reached from an assigned pixel
    @inbounds for p in lat.missing_pad
        Y[p] == 0x00 && (Y[p] = rand(rng, 1:q))
    end
    return Y
end

function _mode_label(counts::Vector{Int}, rng::AbstractRNG)
    m = maximum(counts)
    nmodes = count(==(m), counts)
    r = rand(rng, 1:nmodes)
    @inbounds for k in eachindex(counts)
        if counts[k] == m
            r -= 1
            r == 0 && return k
        end
    end
    return length(counts) # unreachable
end

function _boltzmann_label(counts::Vector{Int}, β::Real, rng::AbstractRNG)
    s = 0.0
    @inbounds for k in eachindex(counts)
        s += exp(β * counts[k])
    end
    t = rand(rng) * s
    @inbounds for k in eachindex(counts)
        t -= exp(β * counts[k])
        t < 0 && return k
    end
    return length(counts)
end

"""
	initializepotts(nrow::Integer, ncol::Integer, q::Integer, β; rng = Random.default_rng())
Initial labels for marginal simulation: uniformly random at high temperature (β < β_crit), and random blocks with
5% random perturbations at low temperature.
"""
function initializepotts(nrow::Integer, ncol::Integer, q::Integer, β; rng::AbstractRNG = Random.default_rng())
    if β < βcrit(q)
        grid = rand(rng, 1:q, nrow, ncol)
    else
        cluster_size = max(1, min(nrow, ncol) ÷ 4)
        base_grid = rand(rng, 1:q, ceil(Int, nrow / cluster_size), ceil(Int, ncol / cluster_size))
        grid = repeat(base_grid, inner = (cluster_size, cluster_size))[1:nrow, 1:ncol]
        for i = 1:nrow, j = 1:ncol
            if rand(rng) < 0.05
                grid[i, j] = rand(rng, 1:q)
            end
        end
    end
    return grid
end

# ---- Running chains and extracting output ----

"""
	run!(chain, lattice, params, niterations; sw_every = 1, ntasks = 1)
	run!(chains::PottsChains, lattice, params, niterations; sw_every = 1)
Runs `niterations` MCMC iterations of `chain` (initialising it first with [`initialize!`](@ref) if needed). Each
iteration is a chequerboard heat-bath Gibbs sweep, preceded by a Swendsen–Wang sweep if the chain's iteration
counter is a multiple of `sw_every` (`sw_every = 0`: Gibbs only).

The sweeps are parallelised over the blocks of the lattice with `ntasks` tasks (see [`PottsLattice`](@ref)), which
are started once per call and synchronise between the phases of each sweep. This pays off for grids of roughly
64 × 64 pixels or more. The results do not depend on `ntasks`. Keep the default, `ntasks = 1`, when several chains
are already run in parallel.

The parameters may change between calls: build a new [`PottsParams`](@ref) for the same lattice, and the chain
continues from its current state. This is useful, for example, between the iterations of a Monte Carlo EM algorithm
([Wei and Tanner, 1990](https://doi.org/10.1080/01621459.1990.10474930)), such as the neural EM algorithm
([Sainsbury-Dale et al., 2025](https://arxiv.org/abs/2501.04330)).

The chains of a [`PottsChains`](@ref) all run together on their device, and `ntasks` does not apply.
"""
function run!(chain::PottsChain, lat::PottsLattice, par::PottsParams, niterations::Integer; sw_every::Integer = 1, ntasks::Integer = 1)
    _checkchain(chain, lat, par)
    sw_every >= 0 || throw(ArgumentError("sw_every must be non-negative; got $sw_every"))
    ntasks >= 1 || throw(ArgumentError("ntasks must be positive; got $ntasks"))
    ws = _workspace()
    chain.initialized || _initialize!(chain, lat, par, ws)
    return _run!(chain, lat, par, ws, niterations, sw_every, Int(ntasks), Val(par.q))
end

function _run!(chain::PottsChain, lat::PottsLattice, par::PottsParams, ws::PottsWorkspace, niterations::Integer, sw_every::Integer, ntasks::Int, ::Val{Q}) where {Q}
    ntasks = min(ntasks, lat.nblocks)
    ntasks > 1 && return _run_parallel!(chain, lat, par, ws, niterations, sw_every, ntasks, Val(Q))
    for _ = 1:niterations
        chain.iter += 1
        if sw_every > 0 && chain.iter % sw_every == 0
            sw_sweep!(chain, lat, par, ws, Val(Q))
        end
        gibbs_sweep!(chain, lat, par, ws, Val(Q))
    end
    return chain
end

# _run! with ntasks > 1 persistent tasks, which share the blocks of each phase of a sweep (handed out by an atomic
# counter) and wait for each other at a barrier at the end of each phase. The last task to arrive at a barrier runs the
# sequential steps: the merge across blocks of a Swendsen–Wang sweep, and resetting the counter. The tasks are started
# once per call, rather than for every phase, and the scratch arrays are sized before they start.
function _run_parallel!(chain::PottsChain, lat::PottsLattice, par::PottsParams, ws::PottsWorkspace, niterations::Integer, sw_every::Integer, ntasks::Int, ::Val{Q}) where {Q}
    niterations > 0 || return chain
    Y = chain.Y
    parent, bonded, clw, forced, newlab, u = _parallel_scratch!(ws, lat, length(Y), sw_every, Val(Q))
    ticket = Threads.Atomic{Int}(0)
    nb = lat.nblocks
    iter0 = chain.iter
    reset = () -> (ticket[] = 0; nothing)
    mergereset = () -> (_sw_merge!(Y, parent, bonded, clw, forced, ws.merged, u, lat, par.pbond, par.logW, Val(Q)); ticket[] = 0; nothing)
    _withworkers(ntasks) do _, barrier
        for it = 1:niterations
            if sw_every > 0 && (iter0 + it) % sw_every == 0
                _phase!(ticket, nb, _sw_block!, Y, parent, bonded, clw, forced, u, chain.blockrngs, lat, par.pbond, par.logW, Val(Q))
                _wait!(mergereset, barrier)
                _phase!(ticket, nb, _sw_rootlabels!, parent, bonded, clw, forced, newlab, u, lat, Val(Q))
                _wait!(reset, barrier)
                _phase!(ticket, nb, _sw_assign!, Y, parent, bonded, newlab, u, lat, par.W, Val(Q))
                _wait!(reset, barrier)
            end
            for c = 1:2
                upd = lat.update[c]
                isempty(upd) && continue
                _phase!(ticket, nb, _gibbs_block!, Y, chain.blockrngs, lat.blockupd[c], upd, lat.wcol[c], par.W, par.E, u, lat.stride, Val(Q))
                _wait!(reset, barrier)
            end
        end
    end
    chain.iter = iter0 + niterations
    return chain
end

# Scratch arrays for _run_parallel!: those of a Swendsen–Wang sweep if it runs any, and uniforms for the Gibbs sweeps
function _parallel_scratch!(ws::PottsWorkspace, lat::PottsLattice, npad::Int, sw_every::Integer, ::Val{Q}) where {Q}
    sw_every > 0 && return _sw_scratch!(ws, lat, npad, Val(Q))
    u = resize!(ws.u, max(length(lat.update[1]), length(lat.update[2])))
    return (ws.parent, ws.bonded, ws.clw, ws.forced, ws.newlab, u)
end

"""
	getlabels!(Yout::AbstractMatrix, chain, lattice)
	getlabels!(Yout::AbstractArray{<:Integer, 3}, chains::PottsChains, lattice)
Copies the current labels of `chain` (integers in `1:q`) into the `nrow`×`ncol` matrix `Yout`, and returns `Yout`.
For chains on a device ([`PottsChains`](@ref)), `Yout` is an `nrow`×`ncol`×`nchains` array on the same device.
"""
function getlabels!(Yout::AbstractMatrix, chain::PottsChain, lat::PottsLattice)
    _checkchain(chain, lat)
    size(Yout) == (lat.nrow, lat.ncol) || throw(DimensionMismatch("Yout must be $(lat.nrow)×$(lat.ncol); got $(size(Yout))"))
    return _getlabels!(Yout, chain, lat)
end

function _getlabels!(Yout::AbstractMatrix, chain::PottsChain, lat::PottsLattice)
    Y = chain.Y
    @inbounds for j = 1:lat.ncol, i = 1:lat.nrow
        Yout[i, j] = Y[_padindex(i, j, lat.stride)]
    end
    return Yout
end

"""
	drawZ!(Zout::AbstractMatrix, chain, lattice, params)
	drawZ!(Zout::AbstractArray{<:Real, 3}, chains::PottsChains, lattice, params)
Draws the missing entries of `Zout` (an `nrow`×`ncol` array whose observed entries are already filled) from the
emission distributions given the current labels of `chain`, and returns `Zout`. The other entries are not modified.

For chains on a device ([`PottsChains`](@ref)), `Zout` is an `nrow`×`ncol`×`nchains` array on the same device. The
data are drawn on the device when every emission distribution is a `Normal`, `LogNormal`, `Uniform`, `Exponential`,
`Gamma`, `Beta` or `Dirac` distribution and `Zout` has a floating-point element type. Otherwise, the labels are
copied to the CPU, the data are drawn there and then copied back.
"""
function drawZ!(Zout::AbstractMatrix, chain::PottsChain, lat::PottsLattice, par::PottsParams)
    _checkchain(chain, lat, par)
    size(Zout) == (lat.nrow, lat.ncol) || throw(DimensionMismatch("Zout must be $(lat.nrow)×$(lat.ncol); got $(size(Zout))"))
    isnothing(par.samplers) && throw(ArgumentError("drawZ! needs parameters built from emission distributions"))
    return _drawZ!(Zout, chain.Y, chain.rng, lat.missing_lin, lat.missing_pad, par.samplers)
end

function _drawZ!(Zout, Y, rng, missing_lin, missing_pad, samplers)
    @inbounds for t in eachindex(missing_lin)
        Zout[missing_lin[t]] = rand(rng, samplers[Y[missing_pad[t]]])
    end
    return Zout
end

# Smallest grid (number of pixels) for which the sweeps of a chain are parallelised over several tasks: below it,
# synchronising the tasks costs more than it saves (one chain of 64 × 64 pixels is about 2× faster on several threads,
# with or without Swendsen–Wang updates; see docs/src/performance.md)
const _MIN_PIXELS_PARALLEL = 64 * 64

# Tasks per chain and tasks over chains: the chains are distributed over min(ntasks, nchains) tasks, and the remaining
# tasks parallelise the sweeps within each chain (if the grid is large enough)
function _splittasks(lat::PottsLattice, ntasks::Integer, nchains::Integer)
    outer = max(1, min(ntasks, nchains))
    inner = lat.nrow * lat.ncol < _MIN_PIXELS_PARALLEL ? 1 : clamp(ntasks ÷ outer, 1, lat.nblocks)
    return outer, inner
end

# Core multi-chain driver: returns labels (and Z, if zfill is given) as nrow × ncol × length(save_at) × nchains arrays.
# The chains share the parameters `pars` (a PottsParams), or chain c has pars[c]. With a backend, the chains run on
# that device instead (_simulate_device, src/gpu.jl and the extension).
function _simulate(lat::PottsLattice, pars::_ChainParams, nchains::Integer, num_iterations::Integer, sw_every::Integer, save_at, rng::AbstractRNG, zfill; ntasks::Integer = Threads.nthreads(), backend = nothing)
    isnothing(backend) || return _simulate_device(lat, pars, nchains, num_iterations, sw_every, save_at, rng, zfill, backend, ntasks)
    nchains > 0 || throw(ArgumentError("nchains must be positive; got $nchains"))
    pars isa PottsParams || length(pars) == nchains || throw(DimensionMismatch("give one PottsParams per chain"))
    sw_every >= 0 || throw(ArgumentError("sw_every must be non-negative; got $sw_every"))
    ntasks >= 1 || throw(ArgumentError("ntasks must be positive; got $ntasks"))
    save_at = _check_save_at(save_at, num_iterations)
    nsave = length(save_at)
    Yout = Array{Int}(undef, lat.nrow, lat.ncol, nsave, nchains)
    Zout = isnothing(zfill) ? nothing : Array{eltype(zfill)}(undef, lat.nrow, lat.ncol, nsave, nchains)
    seeds = rand(rng, UInt64, nchains)
    outer, inner = _splittasks(lat, ntasks, nchains)
    _tforeach(nchains; ntasks = outer, workspace = PottsWorkspace) do c, ws
        par = _param(pars, c)
        chain = PottsChain(lat, Xoshiro(seeds[c]))
        _initialize!(chain, lat, par, ws)
        for (s, target) in enumerate(save_at)
            _run!(chain, lat, par, ws, target - chain.iter, sw_every, inner, Val(par.q))
            _getlabels!(view(Yout, :, :, s, c), chain, lat)
            if !isnothing(Zout)
                Zv = view(Zout, :, :, s, c)
                copyto!(Zv, zfill)
                _drawZ!(Zv, chain.Y, chain.rng, lat.missing_lin, lat.missing_pad, par.samplers)
            end
        end
    end
    return isnothing(Zout) ? Yout : (Y = Yout, Z = Zout)
end

# ---- Hidden Potts ----

# Element type of data drawn from the emission distributions (of every chain, if they are given per chain)
function _emissiontype(distributions)
    _perchain(distributions) && return reduce(promote_type, (_emissiontype(d) for d in distributions))
    return promote_type(map(eltype, distributions)...)
end

"""
	simulatehiddenpotts(nrow::Integer, ncol::Integer, β, distributions; nchains = 1, num_iterations = 1000, sw_every = 1, save_at = num_iterations, rng = Random.default_rng(), ntasks = Threads.nthreads(), backend = nothing)
	simulatehiddenpotts(Z₁::AbstractMatrix, β, distributions; nchains = 1, num_iterations = 50, sw_every = 1, save_at = num_iterations, rng = Random.default_rng(), ntasks = Threads.nthreads(), backend = nothing)
MCMC sampling from a spatial hidden Potts model with inverse-temperature parameter `β` ≥ 0 and emission
`distributions` (one per label, so that the number of labels is `q = length(distributions)`), either unconditionally
on an `nrow`×`ncol` grid (first method) or conditionally on the observed values of the incomplete data `Z₁`, in which
`missing` marks the values to simulate (second method).

Runs `nchains` independent chains (in parallel over threads) for `num_iterations` iterations each, where an
iteration is a chequerboard heat-bath Gibbs sweep preceded, every `sw_every` iterations, by a Swendsen–Wang
sweep (`sw_every = 0`: Gibbs only). Observed values equal to the atom of a `Dirac` emission distribution are
assigned to that label and held fixed.

Returns a named tuple `(Y, Z)` of labels (integers in `1:q`) and data, each an array of dimension
`nrow`×`ncol`×`length(save_at)`×`nchains` containing the states after the iterations in `save_at` (by default, the
final iteration only; iteration 0 is the initial state). In the conditional case, the observed entries of `Z` equal
those of `Z₁`.

Each chain can have its own parameters, e.g. to simulate a batch of training data with one call: `β` can be a vector
with one value per chain, and `distributions` a vector with one vector of emission distributions per chain (each of
the same length `q`). `nchains` then defaults to the number of chains given, and chain `c` uses `β[c]` and
`distributions[c]`. In conditional simulation, the `Dirac` emission distributions must be the same in every chain,
since they fix the observed values equal to their atoms; on a GPU, emission distributions per chain are supported in
unconditional simulation only.

Up to `ntasks` tasks run in parallel (set `ntasks = 1` when calling from code that is already parallel). The
chains are distributed over `min(ntasks, nchains)` tasks; when there are fewer chains than tasks, the remaining tasks
parallelise the sweeps of each chain, for grids of roughly 64 × 64 pixels or more.
Results are reproducible given `rng`, from which the chains are seeded, and do not depend on `ntasks` or on the
number of threads.

To run the chains on a GPU, set `backend` to a KernelAbstractions backend, such as `CUDABackend()` after
`using CUDA`. All `nchains` chains then run as one batch on the device, and `Y` and `Z` are arrays on the device (copy
them to the CPU with `Array`). The chains are still initialised on the CPU, over `ntasks` tasks. The results are
reproducible given `rng`, but differ from those of the CPU sampler. See [GPU sampling](@ref).

# Examples
```julia
using HiddenMarkovRandomFields, Distributions

# Inverse temperature and emission distributions (q = 2 labels)
β = 0.8
distributions = [Normal(0, 1), Normal(2, 1)]

# Unconditional simulation
Y, Z = simulatehiddenpotts(100, 100, β, distributions)
z = Z[:, :, 1, 1]

# Conditional simulation, with 30 independent chains
Z₁ = Matrix{Union{Missing, Float64}}(z)
Z₁[rand(size(z)...) .< 0.2] .= missing
Y_imputed, Z_imputed = simulatehiddenpotts(Z₁, β, distributions; nchains = 30)

# Store the chains after selected iterations
Y_trace, Z_trace = simulatehiddenpotts(Z₁, β, distributions; num_iterations = 100, save_at = [1, 10, 50, 100])

# 500 fields, each with its own β and emission standard deviation (e.g., training data)
βs = rand(500)
σs = rand(500)
Y_train, Z_train = simulatehiddenpotts(64, 64, βs, [[Normal(0, σ), Normal(2, σ)] for σ in σs])

# 1000 chains on an NVIDIA GPU: Y and Z are CuArrays
using CUDA
Y_gpu, Z_gpu = simulatehiddenpotts(Z₁, β, distributions; nchains = 1000, backend = CUDABackend())
```
"""
function simulatehiddenpotts(
    Z₁::AbstractMatrix{<:Union{Missing, Real}}, β, distributions;
    nchains::Integer = _nchains(β, distributions),
    num_iterations::Integer = 50,
    sw_every::Integer = 1,
    save_at = num_iterations,
    rng::AbstractRNG = Random.default_rng(),
    ntasks::Integer = Threads.nthreads(),
    backend = nothing
)
    lat = PottsLattice(Z₁, _latticedistributions(distributions))
    pars = _chainparams(lat, β, distributions, nchains)
    F = nonmissingtype(eltype(Z₁))
    F = isconcretetype(F) ? F : _emissiontype(distributions) # e.g., every value missing
    zfill = convert(Matrix{F}, coalesce.(Z₁, zero(F)))
    return _simulate(lat, pars, nchains, num_iterations, sw_every, save_at, rng, zfill; ntasks = ntasks, backend = backend)
end

function simulatehiddenpotts(
    nrow::Integer, ncol::Integer, β, distributions;
    nchains::Integer = _nchains(β, distributions),
    num_iterations::Integer = 1000,
    sw_every::Integer = 1,
    save_at = num_iterations,
    rng::AbstractRNG = Random.default_rng(),
    ntasks::Integer = Threads.nthreads(),
    backend = nothing
)
    lat = PottsLattice(nrow, ncol)
    pars = _chainparams(lat, β, distributions, nchains)
    zfill = zeros(_emissiontype(distributions), nrow, ncol)
    return _simulate(lat, pars, nchains, num_iterations, sw_every, save_at, rng, zfill; ntasks = ntasks, backend = backend)
end

# ---- Potts ----

"""
	simulatepotts(nrow::Integer, ncol::Integer, q::Integer, β; nchains = 1, num_iterations = 1000, sw_every = 1, save_at = num_iterations, rng = Random.default_rng(), ntasks = Threads.nthreads(), backend = nothing)
	simulatepotts(Y₁::AbstractMatrix, q::Integer, β; nchains = 1, num_iterations = 50, sw_every = 1, save_at = num_iterations, rng = Random.default_rng(), ntasks = Threads.nthreads(), backend = nothing)
MCMC sampling from a spatial `q`-state Potts model with inverse-temperature parameter `β` ≥ 0, either unconditionally
on an `nrow`×`ncol` grid (first method) or conditionally on the observed labels of `Y₁`, in which `missing` marks
the labels to simulate (second method). Labels are integers in `1:q`.

The sampler, keyword arguments (including `backend`, for a GPU) and output layout are those of
[`simulatehiddenpotts`](@ref): returns an array of labels of dimension `nrow`×`ncol`×`length(save_at)`×`nchains`. In the
conditional case, the observed labels are held fixed. As there, `β` can be a vector with one value per chain.

# Examples
```julia
using HiddenMarkovRandomFields

# Unconditional simulation
β = 0.8
Y = simulatepotts(100, 100, 3, β)[:, :, 1, 1]

# Conditional simulation, with 10 independent chains
Y₁ = Matrix{Union{Missing, Int}}(Y)
Y₁[rand(size(Y)...) .< 0.1] .= missing
Y_imputed = simulatepotts(Y₁, 3, β; nchains = 10)

# One field for each of several values of β
Y_β = simulatepotts(100, 100, 3, [0.5, 1.0, 1.5])
```
"""
function simulatepotts(
    nrow::Integer, ncol::Integer, q::Integer, β;
    nchains::Integer = _nchains(β),
    num_iterations::Integer = 1000,
    sw_every::Integer = 1,
    save_at = num_iterations,
    rng::AbstractRNG = Random.default_rng(),
    ntasks::Integer = Threads.nthreads(),
    backend = nothing
)
    lat = PottsLattice(nrow, ncol)
    pars = _chainparams(lat, β, q, nchains)
    return _simulate(lat, pars, nchains, num_iterations, sw_every, save_at, rng, nothing; ntasks = ntasks, backend = backend)
end

function simulatepotts(
    Y₁::AbstractMatrix{<:Union{Missing, Integer}}, q::Integer, β;
    nchains::Integer = _nchains(β),
    num_iterations::Integer = 50,
    sw_every::Integer = 1,
    save_at = num_iterations,
    rng::AbstractRNG = Random.default_rng(),
    ntasks::Integer = Threads.nthreads(),
    backend = nothing
)
    for y in skipmissing(Y₁)
        1 <= y <= q || throw(ArgumentError("the observed labels must be in 1:q = 1:$q; got $y"))
    end
    lat = PottsLattice(ismissing.(Y₁), coalesce.(Y₁, 0), zeros(size(Y₁)))
    pars = _chainparams(lat, β, q, nchains)
    return _simulate(lat, pars, nchains, num_iterations, sw_every, save_at, rng, nothing; ntasks = ntasks, backend = backend)
end
