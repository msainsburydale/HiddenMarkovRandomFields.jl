# Sampling on a GPU. This file holds the persistent device chains (PottsChains), and the code shared with the
# KernelAbstractions extension (ext/HiddenMarkovRandomFieldsKernelAbstractionsExt.jl), which holds the kernels:
#   - counter-based random-number generators (Philox), and division by multiplication;
#   - the layout of the likelihood weights for the Gibbs kernels, and the fixed-point log-weights of the
#     Swendsen–Wang kernels;
#   - samplers of the emission distributions that run on the device.
# None of it needs KernelAbstractions, so it is tested on the CPU. Loading a GPU package (e.g., `using CUDA`) loads
# KernelAbstractions and Atomix, and with them the extension.

# Largest number of labels on a device (the impossible labels of a cluster are the bits of a UInt32)
const _MAXQ_DEVICE = 32

"""
	PottsChains(lattice, nchains::Integer, backend; rng = Random.default_rng())
State of `nchains` MCMC chains on `lattice`, held on the device of the KernelAbstractions `backend`, such as
`CUDABackend()` after `using CUDA`. It contains the labels of every chain, scratch memory for Swendsen–Wang sweeps,
and a copy of the parameter-dependent quantities of the last [`PottsParams`](@ref) used. The scratch memory is
allocated at the first Swendsen–Wang sweep, and takes about 14 + 8q bytes per pixel and chain.

The chains are used like [`PottsChain`](@ref), with [`initialize!`](@ref), [`run!`](@ref), [`getlabels!`](@ref)
and [`drawZ!`](@ref), but all of them are updated together. The arrays given to `getlabels!` and `drawZ!` are
`nrow`×`ncol`×`nchains` arrays on the same device, so the data can be used there, e.g., by a neural network. Pass the
same `lattice` object to every call. The parameters are one `PottsParams` for all chains, or a vector with one per
chain; the chains must then share their likelihood weights, as they always do in unconditional simulation.

Random numbers come from counter-based generators (Philox2x32-10 and Philox4x32-10), seeded from `rng`. Results are
therefore reproducible given `rng`, but differ from those of the CPU sampler.

This needs a GPU package that loads KernelAbstractions, such as CUDA.jl. `KernelAbstractions.CPU()` also works, for
testing. At most 32 labels are supported, and `nchains × (nrow + 2) × (ncol + 2)` must be less than 2³¹.
"""
mutable struct PottsChains{B}
    const backend::B
    const nchains::Int
    const lattice::PottsLattice
    const state::Any   # device arrays and cached parameters (defined by the extension)
    const rng::Xoshiro # initialisation, and data drawn on the CPU for distributions without a device sampler
    const key::UInt32  # Philox key
    const salt::UInt32 # offset of the launch counter, which is the second word of each Philox counter
    launch::UInt32     # number of kernel launches so far that used random numbers
    iter::Int
    initialized::Bool
end

PottsChains(lat::PottsLattice, nchains::Integer, backend; rng::AbstractRNG = Random.default_rng()) = _nodevice(backend)

# The device paths, without the extension (or with a backend that is not a KernelAbstractions backend)
_simulate_device(lat, par, nchains, num_iterations, sw_every, save_at, rng, zfill, backend, ntasks) = _nodevice(backend)

function _nodevice(backend)
    throw(ArgumentError("sampling on a device needs a KernelAbstractions backend, such as `CUDABackend()` from CUDA.jl, " *
                        "and a GPU package must be loaded first (e.g., `using CUDA`); got backend = $(repr(backend))"))
end

function _checkdevice(chains::PottsChains, lat::PottsLattice)
    lat === chains.lattice || throw(ArgumentError("the chains were created for a different lattice: pass the same PottsLattice object"))
    return nothing
end

function _checkdevice(chains::PottsChains, lat::PottsLattice, par::PottsParams)
    _checkdevice(chains, lat)
    size(par.W, 2) == length(lat.obs_lin) + 1 || throw(DimensionMismatch("the parameters were not computed for this lattice"))
    par.q <= _MAXQ_DEVICE || throw(ArgumentError("sampling on a device supports at most $_MAXQ_DEVICE labels; got q = $(par.q)"))
    return nothing
end

# Parameters per chain: one PottsParams for each chain, all with the same number of labels
function _checkdevice(chains::PottsChains, lat::PottsLattice, pars::AbstractVector{<:PottsParams})
    length(pars) == chains.nchains || throw(DimensionMismatch("give one PottsParams per chain: there are $(chains.nchains) chains, but $(length(pars)) PottsParams"))
    for par in pars
        _checkdevice(chains, lat, par)
        par.q == _q(pars) || throw(ArgumentError("every chain must have the same number of labels"))
    end
    return nothing
end

# ---- Random numbers ----

# Philox2x32-10 (Salmon et al., 2011): the two output words for the counter (c0, c1) under `key`
@inline function _philox2x32(key::UInt32, c0::UInt32, c1::UInt32)
    for _ = 1:10
        prod = widemul(0xd256d193, c0)
        c0, c1 = ((prod >> 32) % UInt32) ⊻ key ⊻ c1, prod % UInt32
        key += 0x9e3779b9
    end
    return c0, c1
end

# Philox4x32-10: the four output words for the counter (c0, c1, c2, c3) under the key (k0, k1)
@inline function _philox4x32(k0::UInt32, k1::UInt32, c0::UInt32, c1::UInt32, c2::UInt32, c3::UInt32)
    for _ = 1:10
        p0 = widemul(0xd2511f53, c0)
        p1 = widemul(0xcd9e8d57, c2)
        c0, c1, c2, c3 = ((p1 >> 32) % UInt32) ⊻ c1 ⊻ k0, p1 % UInt32, ((p0 >> 32) % UInt32) ⊻ c3 ⊻ k1, p0 % UInt32
        k0 += 0x9e3779b9
        k1 += 0xbb67ae85
    end
    return c0, c1, c2, c3
end

# A uniform on [0, 1) from the top 24 bits of a random word (for the sweeps)
@inline _u01(x::UInt32) = Float32(x >> 8) * 5.9604645f-8

# A uniform on the open interval (0, 1) from a random word (for drawing data)
@inline _open01(::Type{Float64}, x::UInt32) = (Float64(x) + 0.5) * 2.3283064365386963e-10
@inline _open01(::Type{Float32}, x::UInt32) = (Float32(x >> 9) + 0.5f0) * 1.1920929f-7
@inline _open01(::Type{T}, x::UInt32) where {T <: AbstractFloat} = T(_open01(Float64, x))

# ---- Division by multiplication ----

# (m, s) such that g ÷ d == (g * m) >> s for every 0 ≤ g < 2^31, with m < 2^32 (the round-up method of Granlund and
# Montgomery, 1994): a multiplication and a shift in a kernel, instead of an integer division
function _divmagic(d::Integer)
    d >= 1 || throw(ArgumentError("the divisor must be positive"))
    s = 31 + (64 - leading_zeros(UInt64(d - 1))) # 31 + ⌈log₂ d⌉
    m = cld(UInt128(1) << s, UInt128(d))
    return UInt32(m), UInt32(s)
end

@inline _fastdiv(g::Int32, m::UInt32, s::UInt32) = (widemul(g % UInt32, m) >> s) % Int32

# ---- Parameter-dependent quantities on the device ----

# Likelihood weights of the pixels of each chequerboard colour, in their update order: length(lat.update[c]) × q, in
# Float32 (the Gibbs kernels read them contiguously, with no indirection through the weight-table column)
_colourweights(lat::PottsLattice, par::PottsParams) = map(wcol -> Matrix{Float32}(permutedims(par.W[:, wcol])), lat.wcol)

# Log-likelihood weights in fixed point, so that the cluster sums of a Swendsen–Wang sweep can be accumulated with
# integer atomics, whose results do not depend on the order of the additions: LW[k, o] = round(logW[k, o] * 2^s). An
# impossible label (logW = -Inf) is recorded instead in bit k - 1 of mask[o]. The scale 2^s is the largest for which the
# sum of the weights of all observed pixels cannot overflow.
function _fixedpoint_logweights(par::PottsParams)
    logW = par.logW
    q, ncols = size(logW)
    m = 0.0
    for x in logW
        isfinite(x) && (m = max(m, abs(x)))
    end
    bound = max(ncols - 1, 1) * m
    s = bound == 0 ? 100 : floor(Int, 62 - log2(bound))
    s >= -100 || throw(ArgumentError("the log-likelihood weights are too large for sampling on a device (up to $m in absolute value)"))
    s = min(s, 100)
    LW = Matrix{Int64}(undef, q, ncols)
    mask = zeros(UInt32, ncols)
    for o = 1:ncols, k = 1:q
        x = logW[k, o]
        if x == -Inf
            LW[k, o] = 0
            mask[o] |= one(UInt32) << (k - 1)
        else
            LW[k, o] = round(Int64, ldexp(x, s))
        end
    end
    return LW, mask, s
end

# ---- Emission distributions on the device ----

struct _DevNormal{T}
    μ::T
    σ::T
end
struct _DevLogNormal{T}
    μ::T
    σ::T
end
struct _DevUniform{T}
    a::T
    b::T
end
struct _DevExponential{T}
    θ::T
end
struct _DevGamma{T}
    α::T
    θ::T
end
struct _DevBeta{T}
    α::T
    β::T
end
struct _DevDirac{T}
    v::T
end

# Device sampler of a distribution, for data of type T (nothing: not supported on the device)
_devicesampler(d, ::Type) = nothing
_devicesampler(d::Distributions.Normal, ::Type{T}) where {T <: AbstractFloat} = _DevNormal{T}(d.μ, d.σ)
_devicesampler(d::Distributions.LogNormal, ::Type{T}) where {T <: AbstractFloat} = _DevLogNormal{T}(d.μ, d.σ)
_devicesampler(d::Distributions.Uniform, ::Type{T}) where {T <: AbstractFloat} = _DevUniform{T}(d.a, d.b)
_devicesampler(d::Distributions.Exponential, ::Type{T}) where {T <: AbstractFloat} = _DevExponential{T}(d.θ)
_devicesampler(d::Distributions.Gamma, ::Type{T}) where {T <: AbstractFloat} = _DevGamma{T}(d.α, d.θ)
_devicesampler(d::Distributions.Beta, ::Type{T}) where {T <: AbstractFloat} = _DevBeta{T}(d.α, d.β)
_devicesampler(d::Distributions.Dirac, ::Type{T}) where {T <: AbstractFloat} = _DevDirac{T}(d.value)

# Device samplers of the emission distributions of `par`, as a tuple (one per label), or nothing if any of them is not
# supported on the device
function _devicesamplers(par::PottsParams, ::Type{T}) where {T}
    isnothing(par.distributions) && return nothing
    s = map(d -> _devicesampler(d, T), Tuple(par.distributions))
    return any(isnothing, s) ? nothing : s
end

# The same for parameters per chain: for each label, the sampler of every chain if they are all the same, or a vector
# with the sampler of each chain (moved to the device by the extension), which needs the same concrete type in every
# chain; nothing if a label has neither, or a distribution is not supported on the device
function _devicesamplers(pars::AbstractVector{<:PottsParams}, ::Type{T}) where {T}
    any(par -> isnothing(par.distributions), pars) && return nothing
    s = ntuple(_q(pars)) do k
        v = [_devicesampler(par.distributions[k], T) for par in pars]
        any(isnothing, v) && return nothing
        allequal(v) && return first(v)
        return isconcretetype(eltype(v)) ? v : nothing
    end
    return any(isnothing, s) ? nothing : s
end

# Uniforms for a data value: the n-th group of four, for the counter (c0, c1), under the key (k0, k1)
@inline function _uniforms(::Type{T}, key::NTuple{2, UInt32}, c0::UInt32, c1::UInt32, n::UInt32) where {T}
    w = _philox4x32(key[1], key[2], c0, c1, n, 0x5a5a5a5a)
    return _open01(T, w[1]), _open01(T, w[2]), _open01(T, w[3]), _open01(T, w[4])
end

@inline _boxmuller(u1::T, u2::T) where {T} = sqrt(-2 * log(u1)) * cos(T(2π) * u2)

@inline _draw(d::_DevDirac, key, c0, c1) = d.v

@inline function _draw(d::_DevNormal{T}, key, c0, c1) where {T}
    u = _uniforms(T, key, c0, c1, 0x00000000)
    return d.μ + d.σ * _boxmuller(u[1], u[2])
end

@inline _draw(d::_DevLogNormal{T}, key, c0, c1) where {T} = exp(_draw(_DevNormal{T}(d.μ, d.σ), key, c0, c1))

@inline function _draw(d::_DevUniform{T}, key, c0, c1) where {T}
    u = _uniforms(T, key, c0, c1, 0x00000000)
    return d.a + (d.b - d.a) * u[1]
end

@inline _draw(d::_DevExponential{T}, key, c0, c1) where {T} = -d.θ * log(_uniforms(T, key, c0, c1, 0x00000000)[1])

@inline _draw(d::_DevGamma{T}, key, c0, c1) where {T} = d.θ * exp(first(_loggamma(d.α, key, c0, c1, 0x00000000)))

# Beta(α, β) as Gα / (Gα + Gβ), from the logarithms of the Gamma variates (so that small shapes do not underflow)
@inline function _draw(d::_DevBeta{T}, key, c0, c1) where {T}
    la, n = _loggamma(d.α, key, c0, c1, 0x00000000)
    lb, _ = _loggamma(d.β, key, c0, c1, n)
    return inv(one(T) + exp(lb - la))
end

# Logarithm of a Gamma(α, 1) variate (Marsaglia and Tsang, 2000; for α < 1, the variate for α + 1 times U^(1/α)),
# from the groups of uniforms n, n + 1, …; also returns the index of the next unused group
@inline function _loggamma(α::T, key, c0, c1, n::UInt32) where {T}
    a = α < one(T) ? α + one(T) : α
    d = a - T(1 / 3)
    c = inv(sqrt(9 * d))
    while true
        u = _uniforms(T, key, c0, c1, n)
        n += 0x00000001
        x = _boxmuller(u[1], u[2])
        v = one(T) + c * x
        if v > zero(T)
            v = v * v * v
            if log(u[3]) < x * x / 2 + d - d * v + d * log(v)
                lg = log(d * v)
                if α < one(T)
                    lg += log(u[4]) / α
                end
                return lg, n
            end
        end
    end
end

# The sampler of chain c (0-based) of a label: the same for every chain, or one in a vector per chain
@inline _chainsampler(s, _) = s
@inline _chainsampler(s::AbstractVector, c) = @inbounds s[c + 1]

# A draw from the distribution of label k in chain c (unrolled over the tuple, so that it compiles for a GPU)
@inline _drawlabel(samplers::Tuple{Any}, k, c, key, c0, c1) = _draw(_chainsampler(samplers[1], c), key, c0, c1)
@inline _drawlabel(samplers::Tuple, k, c, key, c0, c1) = k == 1 ? _draw(_chainsampler(samplers[1], c), key, c0, c1) : _drawlabel(Base.tail(samplers), k - 1, c, key, c0, c1)
