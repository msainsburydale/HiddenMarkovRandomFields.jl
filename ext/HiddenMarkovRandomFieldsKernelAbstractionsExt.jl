# Sampling on a GPU with KernelAbstractions: the methods of PottsChains, and the kernels of the chequerboard Gibbs and
# Swendsen–Wang sweeps, of drawing data, and of extracting labels. Loaded with KernelAbstractions and Atomix, e.g., by
# `using CUDA`.
#
# Layout. The labels of all chains are stored pixel by pixel, with the chains of each pixel next to each other: the
# label of chain c (0-based) at padded index p is Y[e] with e = (p - 1) * nchains + c + 1. The threads of a warp
# therefore read and write contiguous bytes, and threads that update the same pixel of different chains share its
# weights. With one chain, consecutive threads take consecutive pixels.
#
# Random numbers. Every random number comes from Philox2x32-10 (Philox4x32-10 for data), with a counter made of an
# item, which identifies the pixel and chain, and the launch counter, which every launch that uses random numbers
# increments. Results are therefore reproducible, and do not depend on the launch configuration.
#
# Swendsen–Wang. Four launches per sweep, each over every label of every chain: (1) bonds are opened and joined by
# lock-free union-find (the root of every cluster is its smallest index); (2) every bonded label points to its root,
# and the fixed-point log-weights, impossible labels and forced labels of each cluster are accumulated at its root with
# atomics, each thread combining a run of labels of its chain first; (3) each cluster, and each singleton, draws its
# new label; (4) bonded labels take their cluster's new label. Integer sums do not depend on the order of the
# additions, so the sweep is deterministic.

module HiddenMarkovRandomFieldsKernelAbstractionsExt

using HiddenMarkovRandomFields
using HiddenMarkovRandomFields: PottsLattice, PottsParams, PottsChain, PottsChains, PottsWorkspace, _initialize!,
    _tforeach, _check_save_at, _drawZ!, _padindex, _checkdevice, _divmagic, _fastdiv, _philox2x32, _u01,
    _colourweights, _fixedpoint_logweights, _devicesamplers, _drawlabel, _ChainParams, _param, _q
import HiddenMarkovRandomFields: initialize!, run!, getlabels!, drawZ!, _simulate_device
using KernelAbstractions
using KernelAbstractions: Backend, allocate, synchronize, get_backend
using Atomix: Atomix
using Random: AbstractRNG, Xoshiro, default_rng

# Workgroup size of every kernel (a Ref, so that the tests can check that the results do not depend on it)
const WORKGROUPSIZE = Ref(256)

# ---- Device state ----

# Device arrays of a PottsChains: the labels, the lattice, the parameter-dependent quantities (rebuilt whenever the
# parameters change), and the scratch memory of the Swendsen–Wang sweeps (allocated at the first such sweep)
mutable struct DeviceState{M8 <: AbstractMatrix{UInt8}, V8 <: AbstractVector{UInt8}, V32 <: AbstractVector{Int32}}
    const Y::M8               # labels: nchains × npad
    const upd::NTuple{2, V32} # padded indices of the pixels to update, per chequerboard colour
    const colpad::V32         # weight-table column of every padded index (0: border or fixed pixel; 1: missing)
    const fixlab::V8          # label of every padded index that is fixed (0 otherwise)
    const padindex::V32       # padded index of every pixel, in column-major order
    const missing_pad::V32    # padded indices of the missing pixels
    const missing_lin::V32    # their linear indices in an nrow × ncol array
    params::Any               # DeviceParams built from `source`
    source::Any               # the PottsParams from which `params` was built (a copy of the vector, if one per chain)
    sw::Any                   # SWScratch (nothing before the first Swendsen–Wang sweep)
end

# E and pbond are scalars if every chain has the same β, and otherwise arrays with one value per chain, which the
# kernels index (see _chainE and _chainpbond)
struct DeviceParams{MW, TE, TP, VL, VM}
    W::NTuple{2, MW} # likelihood weights of the pixels of each colour, in update order (Float32, nupd × q)
    E::TE            # exp(β n) for n = 0, …, 4: an NTuple{5, Float32}, or a vector of nchains (see _chainE)
    pbond::TP        # Swendsen–Wang bond probability: a Float32, or a vector of nchains
    LW::VL           # fixed-point log-likelihood weights (Int64, q × (nobs + 1), column-major)
    mask::VM         # impossible labels of each weight-table column (UInt32 bit masks)
    scale::Float32   # the value of one unit of LW
end

# (The sums of each label are in a plane of their own, which spreads the atomic additions to the root of a large cluster
# over the memory: interleaving the labels of each root was 30% slower above the critical value.)
struct SWScratch{V32, V8, V64, VU64}
    parent::V32 # union-find forest: the index in Y of each label's parent (the identity between sweeps)
    bonds::V8   # bonds opened by each label: bit 0 to its lower neighbour, bit 1 to its right neighbour
    sums::V64   # cluster sums of LW at the roots (plane k: label k; zero between sweeps)
    flags::VU64 # impossible labels of each cluster (bits 0–31), and its forced label (bits 32–39; zero between sweeps)
    newlab::V8  # new label of each cluster, at its root
end

function _todevice(backend, x::AbstractArray)
    y = allocate(backend, eltype(x), size(x))
    isempty(x) || copyto!(y, x)
    return y
end

function _checkarray(A, chains::PottsChains, name)
    typeof(get_backend(A)) == typeof(chains.backend) || throw(ArgumentError("$name must be an array on the device of the chains ($(chains.backend)); got a $(typeof(A))"))
    return nothing
end

# The second word of the next Philox counter
@inline function _nextlaunch!(chains::PottsChains)
    chains.launch += one(UInt32)
    return chains.salt + chains.launch
end

function HiddenMarkovRandomFields.PottsChains(lat::PottsLattice, nchains::Integer, backend::Backend; rng::AbstractRNG = default_rng())
    nchains >= 1 || throw(ArgumentError("nchains must be positive; got $nchains"))
    npad = lat.stride * (lat.ncol + 2)
    nchains * npad < typemax(Int32) || throw(ArgumentError("too many chains for one batch: nchains × (nrow + 2) × (ncol + 2) must be less than 2^31, " *
                                                           "but there are $nchains chains of $(lat.nrow) × $(lat.ncol) pixels; split them into several batches"))
    padindex = Int32[_padindex(i, j, lat.stride) for j = 1:lat.ncol for i = 1:lat.nrow]
    state = DeviceState(KernelAbstractions.zeros(backend, UInt8, nchains, npad), map(u -> _todevice(backend, u), lat.update),
        _todevice(backend, lat.col_pad), _todevice(backend, lat.fixlab_pad), _todevice(backend, padindex),
        _todevice(backend, lat.missing_pad), _todevice(backend, Int32.(lat.missing_lin)), nothing, nothing, nothing)
    key, salt = rand(rng, UInt32), rand(rng, UInt32)
    return PottsChains(backend, Int(nchains), lat, state, Xoshiro(rand(rng, UInt64)), key, salt, zero(UInt32), 0, false)
end

# The quantities that depend on the parameters, on the device (rebuilt only when the parameters change)
function _deviceparams!(chains::PottsChains, st::DeviceState, par::PottsParams)
    st.source === par && return st.params
    backend = chains.backend
    LW, mask, s = _fixedpoint_logweights(par)
    W = map(w -> _todevice(backend, w), _colourweights(chains.lattice, par))
    st.params = DeviceParams(W, map(Float32, par.E), Float32(par.pbond), _todevice(backend, vec(LW)), _todevice(backend, mask), Float32(exp2(-s)))
    st.source = par
    return st.params
end

# The same for parameters per chain. The chains must share the likelihood weights (and so differ only in β): this
# always holds in unconditional simulation, where every weight is 1. With one β for all chains, E and pbond are
# scalars, as above, unless SHARED_WHEN_EQUAL[] is false (a Ref, so that the tests can check that the kernels give the
# same results with arrays of equal values).
const SHARED_WHEN_EQUAL = Ref(true)

function _deviceparams!(chains::PottsChains, st::DeviceState, pars::AbstractVector{<:PottsParams})
    source = st.source
    source isa Vector && length(source) == length(pars) && all(i -> source[i] === pars[i], eachindex(pars)) && return st.params
    par = first(pars)
    all(p -> p.W === par.W || p.W == par.W, pars) ||
        throw(ArgumentError("on a device, the chains must share their likelihood weights: emission distributions per chain are supported only in unconditional simulation"))
    backend = chains.backend
    LW, mask, s = _fixedpoint_logweights(par)
    W = map(w -> _todevice(backend, w), _colourweights(chains.lattice, par))
    if SHARED_WHEN_EQUAL[] && all(p -> p.β == par.β, pars)
        E, pbond = map(Float32, par.E), Float32(par.pbond)
    else
        E, pbond = _todevice(backend, [_chainEvalues(p.E) for p in pars]), _todevice(backend, Float32[p.pbond for p in pars])
    end
    st.params = DeviceParams(W, E, pbond, _todevice(backend, vec(LW)), _todevice(backend, mask), Float32(exp2(-s)))
    st.source = collect(pars) # the cached quantities hold while the chains have these PottsParams
    return st.params
end

function _swscratch!(chains::PottsChains, st::DeviceState, q::Integer)
    nelem = length(st.Y)
    sw = st.sw
    if isnothing(sw) || length(sw.sums) < q * nelem
        backend = chains.backend
        parent = allocate(backend, Int32, nelem)
        _iota!(backend, WORKGROUPSIZE[])(parent; ndrange = nelem)
        sw = SWScratch(parent, KernelAbstractions.zeros(backend, UInt8, nelem), KernelAbstractions.zeros(backend, Int64, q * nelem),
            KernelAbstractions.zeros(backend, UInt64, nelem), KernelAbstractions.zeros(backend, UInt8, nelem))
        st.sw = sw
    end
    return sw
end

@kernel function _iota!(a)
    i = @index(Global, Linear)
    @inbounds a[i] = i % Int32
end

# ---- Initialisation (on the CPU, in parallel over chains) ----

function _initialize_device!(chains::PottsChains, lat::PottsLattice, pars::_ChainParams, ntasks::Integer)
    nchains = chains.nchains
    Ypad = Matrix{UInt8}(undef, lat.stride * (lat.ncol + 2), nchains)
    seeds = rand(chains.rng, UInt64, nchains)
    _tforeach(nchains; ntasks = ntasks, workspace = PottsWorkspace) do c, ws
        chain = PottsChain(lat, Xoshiro(seeds[c]))
        _initialize!(chain, lat, _param(pars, c), ws)
        copyto!(view(Ypad, :, c), vec(chain.Y))
        return nothing
    end
    copyto!(chains.state.Y, permutedims(Ypad))
    chains.iter = 0
    chains.initialized = true
    return chains
end

# ---- Shared device functions ----

# Neighbour counts of the labels 1, …, Q, from the labels of the four neighbours (0: border): 4-bit fields of one
# integer for Q ≤ 15, comparisons otherwise
@inline function _counts(a::UInt8, b::UInt8, l::UInt8, r::UInt8, ::Val{Q}) where {Q}
    if Q <= 7
        nb = Base.shl_int(one(UInt32), UInt32(a) << 2) + Base.shl_int(one(UInt32), UInt32(b) << 2) +
             Base.shl_int(one(UInt32), UInt32(l) << 2) + Base.shl_int(one(UInt32), UInt32(r) << 2)
        return ntuple(k -> ((nb >> (4k)) & 0x0000000f) % Int32, Val(Q))
    elseif Q <= 15
        nb64 = Base.shl_int(one(UInt64), UInt64(a) << 2) + Base.shl_int(one(UInt64), UInt64(b) << 2) +
               Base.shl_int(one(UInt64), UInt64(l) << 2) + Base.shl_int(one(UInt64), UInt64(r) << 2)
        return ntuple(k -> ((nb64 >> (4k)) & 0x000000000000000f) % Int32, Val(Q))
    else
        return ntuple(k -> Int32(a == k) + Int32(b == k) + Int32(l == k) + Int32(r == k), Val(Q))
    end
end

# E[n + 1] for n = 0, …, 4, without indexing the tuple at run time (which would put it in local memory)
@inline _pick(E::NTuple{5, Float32}, n::Int32) =
    ifelse(n < Int32(2), ifelse(n == Int32(0), E[1], E[2]), ifelse(n == Int32(2), E[3], ifelse(n == Int32(3), E[4], E[5])))

# Label with probability ∝ w, from the uniform u on [0, 1). (The closures passed to ntuple in the kernels never capture
# a variable that is assigned more than once, which Julia would box, and which a GPU cannot compile; hence the small
# helper functions below.)
@inline function _drawweights(w::NTuple{Q, Float32}, u::Float32) where {Q}
    total = 0.0f0
    for m = 1:Q
        total += w[m]
    end
    s = u * total
    k = Int32(1)
    partial = 0.0f0
    for m = 1:(Q - 1)
        partial += w[m]
        k += Int32(s >= partial)
    end
    return k % UInt8
end

# The likelihood weights of the pixel at position t (0-based) of a colour's update list
@inline _loadweights(W, t::Int32, nupd::Int32, ::Val{Q}) where {Q} = ntuple(k -> @inbounds(W[t + Int32(1) + Int32(k - 1) * nupd]), Val(Q))

# Full-conditional weights: the likelihood weights times exp(β n_k)
@inline _condweights(w::NTuple{Q, Float32}, E::NTuple{5, Float32}, n::NTuple{Q, Int32}) where {Q} = ntuple(k -> w[k] * _pick(E, n[k]), Val(Q))

# The table exp(β n) of chain c (0-based): the same for every chain (the kernels then compile to the same code as
# without this function), or element c + 1 of a vector with exp(β n) for n = 1, …, 4 of each chain (exp(0) = 1), whose
# 16-byte alignment lets a thread load them with one instruction. (Five values per chain, loaded one at a time, made
# Gibbs sweeps of many chains about 14% slower than with one table for all chains; one load, about 4%.)
const _ChainE = NTuple{4, VecElement{Float32}}
_chainEvalues(E::NTuple{5, Float64}) = ntuple(n -> VecElement(Float32(E[n + 1])), Val(4))
@inline _chainE(E::NTuple{5, Float32}, _) = E
@inline function _chainE(E::AbstractVector{_ChainE}, c::Int32)
    v = @inbounds E[c + Int32(1)]
    return (1.0f0, v[1].value, v[2].value, v[3].value, v[4].value)
end

# ---- Gibbs sweep ----

# One chequerboard colour: thread g updates chain g % nchains at position g ÷ nchains of the colour's update list. Its
# uniform is word c % 2 of Philox(t * npairs + c ÷ 2, launch), for chain c at position t.
@kernel function _gibbs_kernel!(Y, @Const(upd), @Const(W), E, nchains::Int32, cstride::Int32, nupd::Int32, npairs::Int32, divm::UInt32, divs::UInt32, key::UInt32, ctr::UInt32, ::Val{Q}) where {Q}
    g = (@index(Global, Linear) - 1) % Int32
    t = _fastdiv(g, divm, divs) # position in the update list (0-based): g ÷ nchains
    c = g - t * nchains         # chain (0-based)
    @inbounds begin
        i = (upd[t + Int32(1)] - Int32(1)) * nchains + c + Int32(1)
        n = _counts(Y[i - nchains], Y[i + nchains], Y[i - cstride], Y[i + cstride], Val(Q))
        w = _condweights(_loadweights(W, t, nupd, Val(Q)), _chainE(E, c), n)
        w0, w1 = _philox2x32(key, (t * npairs + (c >> 1)) % UInt32, ctr)
        Y[i] = _drawweights(w, _u01(iszero(c & Int32(1)) ? w0 : w1))
    end
end

# The same update, with thread g taking the pair of chains 2h and 2h + 1 (h = g % npairs) at position g ÷ npairs: they
# share the pixel's index and weights, and the two words of one Philox call (the same numbers as above)
@kernel function _gibbs_pairs_kernel!(Y, @Const(upd), @Const(W), E, nchains::Int32, cstride::Int32, nupd::Int32, npairs::Int32, divm::UInt32, divs::UInt32, key::UInt32, ctr::UInt32, ::Val{Q}) where {Q}
    g = (@index(Global, Linear) - 1) % Int32
    t = _fastdiv(g, divm, divs) # position in the update list (0-based): g ÷ npairs
    h = g - t * npairs          # pair of chains (0-based)
    @inbounds begin
        i = (upd[t + Int32(1)] - Int32(1)) * nchains + Int32(2) * h + Int32(1)
        w = _loadweights(W, t, nupd, Val(Q))
        w0, w1 = _philox2x32(key, (t * npairs + h) % UInt32, ctr)
        ni = _counts(Y[i - nchains], Y[i + nchains], Y[i - cstride], Y[i + cstride], Val(Q))
        Y[i] = _drawweights(_condweights(w, _chainE(E, Int32(2) * h), ni), _u01(w0))
        if Int32(2) * h + Int32(1) < nchains
            j = i + Int32(1)
            nj = _counts(Y[j - nchains], Y[j + nchains], Y[j - cstride], Y[j + cstride], Val(Q))
            Y[j] = _drawweights(_condweights(w, _chainE(E, Int32(2) * h + Int32(1)), nj), _u01(w1))
        end
    end
end

# Whether a thread updates two chains (a Ref, so that the tests can compare the two kernels)
const GIBBS_PAIRS = Ref(true)

function _gibbs_sweep!(chains::PottsChains, st::DeviceState, dp::DeviceParams, ::Val{Q}) where {Q}
    pairs = GIBBS_PAIRS[]
    kernel! = (pairs ? _gibbs_pairs_kernel! : _gibbs_kernel!)(chains.backend, WORKGROUPSIZE[])
    nchains = Int32(chains.nchains)
    cstride = nchains * Int32(chains.lattice.stride)
    npairs = Int32(cld(chains.nchains, 2))
    nthreads = pairs ? Int(npairs) : chains.nchains # threads per pixel
    divm, divs = _divmagic(nthreads)
    for colour = 1:2
        nupd = length(st.upd[colour])
        nupd == 0 && continue
        kernel!(st.Y, st.upd[colour], dp.W[colour], dp.E, nchains, cstride, Int32(nupd), npairs, divm, divs, chains.key, _nextlaunch!(chains), Val(Q); ndrange = nupd * nthreads)
    end
    return nothing
end

# Several Gibbs sweeps in one launch, for many chains of a grid whose padded labels fit in local memory: workgroup c
# updates chain c - 1, holding its labels in local memory, with a barrier after each half-sweep; its thread l updates
# the positions l - 1, l - 1 + WG, … of each colour's update list. The random numbers are those of the kernels above
# (the launch counter advances by one per nonempty half-sweep), so the results are the same, without a launch per
# half-sweep or the traffic of the labels to and from global memory.
@kernel unsafe_indices = true function _gibbs_fused_kernel!(Y, @Const(upd1), @Const(upd2), @Const(W1), @Const(W2), E, nchains::Int32, stride::Int32, nupd1::Int32, nupd2::Int32, npairs::Int32, nsweeps::Int32, step::Int32, key::UInt32, ctr0::UInt32, ::Val{Q}, ::Val{NPAD}, ::Val{WG}) where {Q, NPAD, WG}
    gi = @index(Group, Linear)
    li = @index(Local, Linear)
    lab = @localmem UInt8 (NPAD,)
    for p = li:WG:NPAD
        @inbounds lab[p] = Y[(p - 1) * Int(nchains) + gi]
    end
    @synchronize
    for s = Int32(0):(nsweeps - Int32(1))
        _fused_halfsweep!(lab, upd1, W1, E, nupd1, stride, npairs, Int32(gi - 1), Int32(li - 1), key, ctr0 + (s * step + Int32(1)) % UInt32, Val(Q), Val(WG))
        @synchronize
        _fused_halfsweep!(lab, upd2, W2, E, nupd2, stride, npairs, Int32(gi - 1), Int32(li - 1), key, ctr0 + (s * step + step) % UInt32, Val(Q), Val(WG))
        @synchronize
    end
    for p = li:WG:NPAD
        @inbounds Y[(p - 1) * Int(nchains) + gi] = lab[p]
    end
end

# (The chain's table exp(β n) is read here, in each half-sweep: on the CPU backend, a value computed in the kernel
# before a @synchronize is not available after it.)
@inline function _fused_halfsweep!(lab, upd, W, E, nupd::Int32, stride::Int32, npairs::Int32, c::Int32, l::Int32, key::UInt32, ctr::UInt32, ::Val{Q}, ::Val{WG}) where {Q, WG}
    Ec = _chainE(E, c)
    t = l
    @inbounds while t < nupd
        p = upd[t + Int32(1)]
        n = _counts(lab[p - Int32(1)], lab[p + Int32(1)], lab[p - stride], lab[p + stride], Val(Q))
        w = _condweights(_loadweights(W, t, nupd, Val(Q)), Ec, n)
        w0, w1 = _philox2x32(key, (t * npairs + (c >> 1)) % UInt32, ctr)
        lab[p] = _drawweights(w, _u01(iszero(c & Int32(1)) ? w0 : w1))
        t += Int32(WG)
    end
    return nothing
end

# When consecutive Gibbs sweeps share a launch: runs of at least FUSED_MINSWEEPS sweeps of a small batch (at most
# FUSED_MAXLABELS labels) of grids of at most 64 × 64 pixels, for which one launch per half-sweep is limited by the
# launch overhead. Larger batches are faster with the kernels above: a workgroup per chain loads and stores the labels
# of its chain with a stride of nchains, and does not share random numbers between chains. (FUSED_GIBBS is a Ref, so
# that the tests can compare the two paths.)
const FUSED_GIBBS = Ref(true)
const FUSED_MINSWEEPS = 4
const FUSED_MAXPAD = 66 * 66
const FUSED_MAXLABELS = 2^17

function _gibbs_sweeps!(chains::PottsChains, st::DeviceState, dp::DeviceParams, nsweeps::Integer, ::Val{Q}) where {Q}
    lat = chains.lattice
    npad = lat.stride * (lat.ncol + 2)
    if FUSED_GIBBS[] && nsweeps >= FUSED_MINSWEEPS && npad <= FUSED_MAXPAD && chains.nchains * npad <= FUSED_MAXLABELS
        nupd1, nupd2 = length(st.upd[1]), length(st.upd[2])
        step = (nupd1 > 0) + (nupd2 > 0)
        step == 0 && return nothing
        ctr0 = chains.salt + chains.launch
        chains.launch += (nsweeps * step) % UInt32
        wg = WORKGROUPSIZE[]
        _gibbs_fused_kernel!(chains.backend, wg)(st.Y, st.upd[1], st.upd[2], dp.W[1], dp.W[2], dp.E, Int32(chains.nchains), Int32(lat.stride), Int32(nupd1), Int32(nupd2),
            Int32(cld(chains.nchains, 2)), Int32(nsweeps), Int32(step), chains.key, ctr0, Val(Q), Val(npad), Val(wg); ndrange = chains.nchains * wg)
    else
        for _ = 1:nsweeps
            _gibbs_sweep!(chains, st, dp, Val(Q))
        end
    end
    return nothing
end

# ---- Swendsen–Wang sweep ----

# Whether the label at index e of Y has an open bond (to its lower or right neighbour, or from its upper or left one)
@inline _bonded(bonds, e::Int32, nchains::Int32, cstride::Int32) =
    @inbounds (bonds[e] != 0x00) | ((bonds[e - nchains] & 0x01) != 0x00) | ((bonds[e - cstride] & 0x02) != 0x00)

# Root of x, halving the path to it. Parents always have smaller indices than their children, so the forest has no
# cycles. The parent of a label only ever changes to a label of its own tree, or of a tree that _union! is joining to it,
# so a parent read while another thread changes it (a stale one) still leads to a label that ends up in the same tree.
# The loads and stores here are therefore plain ones (which a GPU can cache), and only the links in _union! are atomic.
@inline function _find!(parent, x::Int32)
    @inbounds while true
        px = parent[x]
        px == x && return x
        gp = parent[px]
        gp == px && return px
        parent[x] = gp
        x = gp
    end
end

# Joins the trees of a and b: the larger root is linked to the smaller one with an atomic minimum (Playne and Hawick,
# 2018). If the larger one is no longer a root, the minimum may lower the parent of a label that is not a root; its
# previous parent is then joined to the smaller root in turn, so that no link is lost. (A compare-and-swap, linking only
# roots, compiles to a system-scope atomic with some versions of CUDA.jl, which makes this kernel about 50 times slower.)
@inline function _union!(parent, a::Int32, b::Int32)
    while true
        a = _find!(parent, a)
        b = _find!(parent, b)
        a == b && return nothing
        if a < b
            a, b = b, a
        end
        old = first(Atomix.@atomic :monotonic parent[a] min b)
        old == a && return nothing
        a = old
    end
end

# Root of x, once the forest is complete
@inline function _root(parent, x::Int32)
    @inbounds while true
        px = parent[x]
        px == x && return x
        x = px
    end
end

# The bond probability of the chain of the label at index e of Y: the same for every chain, or that of chain
# (e - 1) % nchains in a vector
@inline _chainpbond(pbond::Float32, _, _, _, _) = pbond
@inline function _chainpbond(pbond::AbstractVector{Float32}, e::Int32, nchains::Int32, divm::UInt32, divs::UInt32)
    i = e - Int32(1)
    return @inbounds pbond[i - _fastdiv(i, divm, divs) * nchains + Int32(1)]
end

# 1. Each label opens its bonds to its lower and right neighbours (with probability pbond if they have the same label),
# using both words of Philox(e - 1, launch), and joins the trees of the bonded labels
@kernel function _sw_bonds!(parent, bonds, @Const(Y), pbond, nchains::Int32, cstride::Int32, offset::Int32, divm::UInt32, divs::UInt32, key::UInt32, ctr::UInt32)
    e = offset + (@index(Global, Linear) - 1) % Int32
    @inbounds begin
        y = Y[e]
        if y != 0x00
            down = Y[e + nchains] == y
            right = Y[e + cstride] == y
            if down | right
                w0, w1 = _philox2x32(key, (e - Int32(1)) % UInt32, ctr)
                p = _chainpbond(pbond, e, nchains, divm, divs)
                down &= _u01(w0) < p
                right &= _u01(w1) < p
            end
            bonds[e] = UInt8(down) | (UInt8(right) << 1)
            down && _union!(parent, e, e + nchains)
            right && _union!(parent, e, e + cstride)
        end
    end
end

# acc plus the fixed-point log-weights of one weight-table column (from index base + 1 of LW)
@inline _addweights(acc::NTuple{Q, Int64}, LW, base::Int32) where {Q} = ntuple(k -> acc[k] + @inbounds(LW[base + k]), Val(Q))

# Adds the accumulation of a run of labels of one cluster to its root (an atomic per nonzero sum; the flags only if
# they add a bit, since most members of a large cluster carry the same ones)
@inline function _flush!(sums, flags, root::Int32, acc::NTuple{Q, Int64}, fl::UInt64, nelem::Int) where {Q}
    if root != Int32(0)
        for k = 1:Q
            a = acc[k]
            if a != zero(Int64)
                Atomix.@atomic :monotonic sums[root + (k - 1) * nelem] += a
            end
        end
        if fl != zero(UInt64)
            old = @inbounds flags[root] # a stale value has fewer bits, which costs at most an unnecessary atomic
            if (old & fl) != fl
                Atomix.@atomic :monotonic flags[root] |= fl
            end
        end
    end
    return nothing
end

# 2. Thread g takes `strip` consecutive padded indices of chain g % nchains. Each bonded label among them points to the
# root of its cluster from now on, and its contribution is accumulated at the root (whose accumulators are zero): the
# fixed-point log-weights and impossible labels of an observed pixel, or the label of a fixed pixel, which is forced on
# its cluster. The contributions of consecutive labels with the same root are combined before they are added
# atomically, which reduces contention at the root of a cluster that covers most of the grid.
@kernel function _sw_accumulate!(parent, sums, flags, @Const(Y), @Const(bonds), @Const(colpad), @Const(fixlab), @Const(LW), @Const(mask), nchains::Int32, cstride::Int32, stride::Int32, plast::Int32, strip::Int32, divm::UInt32, divs::UInt32, nelem::Int, ::Val{Q}) where {Q}
    g = (@index(Global, Linear) - 1) % Int32
    s = _fastdiv(g, divm, divs) # strip (0-based): g ÷ nchains
    c = g - s * nchains         # chain (0-based)
    p0 = stride + Int32(1) + s * strip
    root = Int32(0)
    acc = ntuple(_ -> zero(Int64), Val(Q))
    fl = zero(UInt64)
    @inbounds for p = p0:min(p0 + strip - Int32(1), plast)
        e = (p - Int32(1)) * nchains + c + Int32(1)
        if Y[e] != 0x00 && _bonded(bonds, e, nchains, cstride)
            r = _root(parent, e)
            parent[e] = r
            col = colpad[p]
            if col != Int32(1) # a missing pixel contributes nothing
                if r != root
                    _flush!(sums, flags, root, acc, fl, nelem)
                    root = r
                    acc = ntuple(_ -> zero(Int64), Val(Q))
                    fl = zero(UInt64)
                end
                if col == Int32(0)
                    fl |= UInt64(fixlab[p]) << 32
                else
                    acc = _addweights(acc, LW, (col - Int32(1)) * Int32(Q))
                    fl |= UInt64(mask[col])
                end
            end
        end
    end
    _flush!(sums, flags, root, acc, fl, nelem)
end

# Label of a cluster with sums S (in units of `scale`), with probability ∝ exp(S_k - max S) over its possible labels
@inline function _drawcluster(sums, e::Int32, m::UInt32, nelem::Int, scale::Float32, u::Float32, ::Val{Q}) where {Q}
    S = ntuple(k -> @inbounds(sums[e + (k - 1) * nelem]), Val(Q))
    smax = _maxpossible(S, m)
    w = ntuple(k -> (m >> (k - 1)) & one(UInt32) == zero(UInt32) ? exp(Float32(S[k] - smax) * scale) : 0.0f0, Val(Q))
    return _drawweights(w, u)
end

# Largest of the sums S of the labels that are possible (not in the mask m)
@inline function _maxpossible(S::NTuple{Q, Int64}, m::UInt32) where {Q}
    smax = typemin(Int64)
    for k = 1:Q
        if (m >> (k - 1)) & one(UInt32) == zero(UInt32)
            smax = max(smax, S[k])
        end
    end
    return smax
end

# Label of a singleton with weight-table column col, with probability ∝ its likelihood weights
@inline function _drawsingle(LW, mask, col::Int32, scale::Float32, u::Float32, ::Val{Q}) where {Q}
    m = @inbounds mask[col]
    base = (col - Int32(1)) * Q
    w = ntuple(k -> (m >> (k - 1)) & one(UInt32) == zero(UInt32) ? exp(Float32(@inbounds LW[base + k]) * scale) : 0.0f0, Val(Q))
    return _drawweights(w, u)
end

# 3. Each root draws the new label of its cluster (its forced label, if it has one) and clears its accumulators for the
# next sweep, and each singleton draws its own new label from its likelihood weights (uniformly if missing; a fixed one
# is unchanged), using word 0 of Philox(e - 1, launch)
@kernel function _sw_relabel!(Y, newlab, sums, flags, @Const(parent), @Const(bonds), @Const(colpad), @Const(LW), @Const(mask), nchains::Int32, cstride::Int32, offset::Int32, divm::UInt32, divs::UInt32, nelem::Int, scale::Float32, key::UInt32, ctr::UInt32, ::Val{Q}) where {Q}
    e = offset + (@index(Global, Linear) - 1) % Int32
    @inbounds if Y[e] != 0x00
        if _bonded(bonds, e, nchains, cstride)
            if parent[e] == e
                fl = flags[e]
                forced = (fl >> 32) % UInt8
                if forced != 0x00
                    newlab[e] = forced
                else
                    u = _u01(first(_philox2x32(key, (e - Int32(1)) % UInt32, ctr)))
                    newlab[e] = _drawcluster(sums, e, fl % UInt32, nelem, scale, u, Val(Q))
                end
                for k = 1:Q
                    sums[e + (k - 1) * nelem] = zero(Int64)
                end
                flags[e] = zero(UInt64)
            end
        else
            col = colpad[_fastdiv(e - Int32(1), divm, divs) + Int32(1)]
            if col != Int32(0)
                u = _u01(first(_philox2x32(key, (e - Int32(1)) % UInt32, ctr)))
                Y[e] = col == Int32(1) ? (min(unsafe_trunc(Int32, u * Float32(Q)), Int32(Q - 1)) + Int32(1)) % UInt8 : _drawsingle(LW, mask, col, scale, u, Val(Q))
            end
        end
    end
end

# 4. Each bonded label takes the new label of its cluster, and resets its parent for the next sweep
@kernel function _sw_assign!(Y, parent, @Const(newlab), @Const(bonds), nchains::Int32, cstride::Int32, offset::Int32)
    e = offset + (@index(Global, Linear) - 1) % Int32
    @inbounds if Y[e] != 0x00 && _bonded(bonds, e, nchains, cstride)
        Y[e] = newlab[parent[e]]
        parent[e] = e
    end
end

# Labels of one chain taken by each thread of the accumulation. Longer strips combine more additions to the root of a
# large cluster, but leave fewer threads; 4 was the best length, or close to it, for one large chain or many small
# ones, above and below the critical value. (A Ref, so that the tests can check that the results do not depend on it.)
const SW_STRIP = Ref(4)

function _sw_sweep!(chains::PottsChains, st::DeviceState, dp::DeviceParams, sw::SWScratch, ::Val{Q}) where {Q}
    backend, wg, lat = chains.backend, WORKGROUPSIZE[], chains.lattice
    nchains = Int32(chains.nchains)
    stride = Int32(lat.stride)
    cstride = nchains * stride
    offset = stride * nchains + Int32(1)            # index in Y of the first label of padded column 1
    nitems = lat.ncol * lat.stride * chains.nchains # the labels of padded columns 1, …, ncol (border rows included)
    nelem = length(st.Y)
    divm, divs = _divmagic(chains.nchains)
    _sw_bonds!(backend, wg)(sw.parent, sw.bonds, st.Y, dp.pbond, nchains, cstride, offset, divm, divs, chains.key, _nextlaunch!(chains); ndrange = nitems)
    strip = SW_STRIP[]
    nstrips = cld(lat.ncol * lat.stride, strip)
    _sw_accumulate!(backend, wg)(sw.parent, sw.sums, sw.flags, st.Y, sw.bonds, st.colpad, st.fixlab, dp.LW, dp.mask, nchains, cstride, stride,
        Int32((lat.ncol + 1) * lat.stride), Int32(strip), divm, divs, nelem, Val(Q); ndrange = nstrips * chains.nchains)
    _sw_relabel!(backend, wg)(st.Y, sw.newlab, sw.sums, sw.flags, sw.parent, sw.bonds, st.colpad, dp.LW, dp.mask, nchains, cstride, offset, divm, divs,
        nelem, dp.scale, chains.key, _nextlaunch!(chains), Val(Q); ndrange = nitems)
    _sw_assign!(backend, wg)(st.Y, sw.parent, sw.newlab, sw.bonds, nchains, cstride, offset; ndrange = nitems)
    return nothing
end

# ---- Running chains ----

# The Gibbs sweeps between two Swendsen–Wang sweeps are run together (in one launch, if possible), in the same order
function _run_device!(chains::PottsChains, st::DeviceState, dp::DeviceParams, niterations::Integer, sw_every::Integer, ::Val{Q}) where {Q}
    pending = 0
    for _ = 1:niterations
        chains.iter += 1
        if sw_every > 0 && chains.iter % sw_every == 0
            _gibbs_sweeps!(chains, st, dp, pending, Val(Q))
            pending = 0
            _sw_sweep!(chains, st, dp, _swscratch!(chains, st, Q), Val(Q))
        end
        pending += 1
    end
    _gibbs_sweeps!(chains, st, dp, pending, Val(Q))
    return chains
end

# `par` is one PottsParams for every chain, or a vector with one per chain
function initialize!(chains::PottsChains, lat::PottsLattice, par::_ChainParams)
    _checkdevice(chains, lat, par)
    return _initialize_device!(chains, lat, par, Threads.nthreads())
end

function run!(chains::PottsChains, lat::PottsLattice, par::_ChainParams, niterations::Integer; sw_every::Integer = 1)
    _checkdevice(chains, lat, par)
    niterations >= 0 || throw(ArgumentError("niterations must be non-negative; got $niterations"))
    sw_every >= 0 || throw(ArgumentError("sw_every must be non-negative; got $sw_every"))
    st = chains.state
    dp = _deviceparams!(chains, st, par)
    chains.initialized || _initialize_device!(chains, lat, par, Threads.nthreads())
    _run_device!(chains, st, dp, niterations, sw_every, Val(_q(par)))
    synchronize(chains.backend)
    return chains
end

# ---- Labels and data ----

# Thread g writes element g of Yout (nrow × ncol × nchains): the label of chain g ÷ npix at pixel g % npix
@kernel function _getlabels_kernel!(Yout, @Const(Y), @Const(padindex), nchains::Int32, npix::Int32, divm::UInt32, divs::UInt32)
    g = (@index(Global, Linear) - 1) % Int32
    c = _fastdiv(g, divm, divs)
    @inbounds Yout[g + Int32(1)] = Y[(padindex[g - c * npix + Int32(1)] - Int32(1)) * nchains + c + Int32(1)]
end

function getlabels!(Yout::AbstractArray{<:Integer, 3}, chains::PottsChains, lat::PottsLattice)
    _checkdevice(chains, lat)
    size(Yout) == (lat.nrow, lat.ncol, chains.nchains) || throw(DimensionMismatch("Yout must be $(lat.nrow)×$(lat.ncol)×$(chains.nchains); got $(size(Yout))"))
    _checkarray(Yout, chains, "Yout")
    npix = lat.nrow * lat.ncol
    st = chains.state
    _getlabels_kernel!(chains.backend, WORKGROUPSIZE[])(Yout, st.Y, st.padindex, Int32(chains.nchains), Int32(npix), _divmagic(npix)...; ndrange = length(Yout))
    synchronize(chains.backend)
    return Yout
end

# Thread g draws the missing value m = g ÷ nchains of chain c = g % nchains, from Philox4x32 with the counter
# (g, launch, n, ⋅), where n counts the groups of four uniforms used for that value
@kernel function _drawZ_kernel!(Z, @Const(Y), @Const(missing_pad), @Const(missing_lin), samplers, nchains::Int32, npix::Int32, divm::UInt32, divs::UInt32, key::NTuple{2, UInt32}, ctr::UInt32)
    g = (@index(Global, Linear) - 1) % Int32
    m = _fastdiv(g, divm, divs)
    c = g - m * nchains
    @inbounds begin
        k = Y[(missing_pad[m + Int32(1)] - Int32(1)) * nchains + c + Int32(1)]
        Z[missing_lin[m + Int32(1)] + c * npix] = _drawlabel(samplers, k, c, key, g % UInt32, ctr)
    end
end

_hassamplers(par::PottsParams) = !isnothing(par.samplers)
_hassamplers(pars::AbstractVector{<:PottsParams}) = all(_hassamplers, pars)

function drawZ!(Zout::AbstractArray{<:Real, 3}, chains::PottsChains, lat::PottsLattice, par::_ChainParams)
    _checkdevice(chains, lat, par)
    size(Zout) == (lat.nrow, lat.ncol, chains.nchains) || throw(DimensionMismatch("Zout must be $(lat.nrow)×$(lat.ncol)×$(chains.nchains); got $(size(Zout))"))
    _checkarray(Zout, chains, "Zout")
    _hassamplers(par) || throw(ArgumentError("drawZ! needs parameters built from emission distributions"))
    chains.initialized || throw(ArgumentError("the chains have not been initialised: call initialize! or run! first"))
    nmiss = length(lat.missing_lin)
    nmiss == 0 && return Zout
    samplers = _devicesamplers(par, eltype(Zout))
    if isnothing(samplers) # a distribution without a sampler on the device: draw on the CPU
        _drawZ_host!(Zout, chains, lat, par)
    else
        samplers = map(s -> s isa AbstractVector ? _todevice(chains.backend, s) : s, samplers) # samplers per chain
        st = chains.state
        key = (chains.key, chains.salt ⊻ 0x6a09e667)
        _drawZ_kernel!(chains.backend, WORKGROUPSIZE[])(Zout, st.Y, st.missing_pad, st.missing_lin, samplers, Int32(chains.nchains),
            Int32(lat.nrow * lat.ncol), _divmagic(chains.nchains)..., key, _nextlaunch!(chains); ndrange = nmiss * chains.nchains)
        synchronize(chains.backend)
    end
    return Zout
end

function _drawZ_host!(Zout, chains::PottsChains, lat::PottsLattice, par::_ChainParams)
    Ypad = permutedims(Array(chains.state.Y)) # npad × nchains
    Zh = Array(Zout)
    for c = 1:chains.nchains
        Yc = reshape(view(Ypad, :, c), lat.stride, lat.ncol + 2)
        _drawZ!(view(Zh, :, :, c), Yc, chains.rng, lat.missing_lin, lat.missing_pad, _param(par, c).samplers)
    end
    copyto!(Zout, Zh)
    return Zout
end

# ---- simulatehiddenpotts and simulatepotts on a device ----

function _simulate_device(lat::PottsLattice, par::_ChainParams, nchains::Integer, num_iterations::Integer, sw_every::Integer, save_at, rng::AbstractRNG, zfill, backend::Backend, ntasks::Integer)
    nchains > 0 || throw(ArgumentError("nchains must be positive; got $nchains"))
    sw_every >= 0 || throw(ArgumentError("sw_every must be non-negative; got $sw_every"))
    ntasks >= 1 || throw(ArgumentError("ntasks must be positive; got $ntasks"))
    save_at = _check_save_at(save_at, num_iterations)
    chains = PottsChains(lat, nchains, backend; rng = rng)
    _checkdevice(chains, lat, par)
    st = chains.state
    dp = _deviceparams!(chains, st, par) # before the initialisation, so that unsupported parameters fail early
    _initialize_device!(chains, lat, par, ntasks)
    nrow, ncol, nsave = lat.nrow, lat.ncol, length(save_at)
    Yout = allocate(backend, Int, (nrow, ncol, nsave, nchains))
    Ys = allocate(backend, Int, (nrow, ncol, nchains))
    hasZ = !isnothing(zfill)
    Zout = hasZ ? allocate(backend, eltype(zfill), (nrow, ncol, nsave, nchains)) : nothing
    Zs = hasZ ? allocate(backend, eltype(zfill), (nrow, ncol, nchains)) : nothing
    Zobs = hasZ ? _todevice(backend, reshape(zfill, nrow, ncol, 1)) : nothing
    for (s, target) in enumerate(save_at)
        _run_device!(chains, st, dp, target - chains.iter, sw_every, Val(_q(par)))
        getlabels!(Ys, chains, lat)
        view(Yout, :, :, s, :) .= Ys
        if hasZ
            Zs .= Zobs
            drawZ!(Zs, chains, lat, par)
            view(Zout, :, :, s, :) .= Zs
        end
    end
    synchronize(backend)
    return hasZ ? (Y = Yout, Z = Zout) : Yout
end

end
