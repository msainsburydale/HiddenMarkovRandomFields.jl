# Tests of sampling on a device (the KernelAbstractions extension): exact targets for many independent chains, data
# drawn on the device, agreement with the CPU sampler on large grids, reproducibility, the persistent device chains,
# and the helpers in src/gpu.jl. They run on KernelAbstractions' CPU backend, unless the including script defines
# GPU_TEST_BACKEND first (benchmark/gpu/test_cuda.jl runs them on an NVIDIA GPU).

using HiddenMarkovRandomFields, Test, Random, Statistics
using HiddenMarkovRandomFields: PottsLattice, PottsParams, PottsChains, initialize!, run!, getlabels!, drawZ!
using Distributions: Normal, LogNormal, Uniform, Exponential, Gamma, Beta, Dirac, TDist, Poisson, mean, var, std
using KernelAbstractions: KernelAbstractions, get_backend, allocate
using Atomix: Atomix

@isdefined(exactposterior) || include("enumeration.jl")

const BACKEND = @isdefined(GPU_TEST_BACKEND) ? GPU_TEST_BACKEND : KernelAbstractions.CPU()
const HMRF = HiddenMarkovRandomFields
const EXT = Base.get_extension(HMRF, :HiddenMarkovRandomFieldsKernelAbstractionsExt)

# Fraction of like-neighbour pairs of each chain (Y: nrow × ncol × nsave × nchains), at the last saved iteration
likefraction(Y) = [likepairs(view(Y, :, :, size(Y, 3), c)) / (size(Y, 1) * (size(Y, 2) - 1) + (size(Y, 1) - 1) * size(Y, 2)) for c in axes(Y, 4)]

# The kernel's division by a multiplication and a shift is exact for every thread index g < 2^31
function divisionexact(d, rng)
    m, s = HMRF._divmagic(d)
    top = Int(typemax(Int32))
    k = top ÷ d
    gs = [0:min(3d, 10_000); k * d - 1; k * d; top; rand(rng, 0:top, 200)]
    return all(g -> (widemul(g % UInt32, m) >> s) % Int == g ÷ d, gs)
end

@testset "sampling on a device ($(nameof(typeof(BACKEND))))" begin
    n = 20_000 # independent chains in the exact-target tests

    @testset "exact targets: hidden Potts, Normal emissions with missing values" begin
        distributions = [Normal(-1.0, 0.6), Normal(0.0, 0.6), Normal(1.0, 0.6)]
        Z = Matrix{Union{Missing, Float64}}(randn(Xoshiro(11), 3, 3) .* 0.8)
        Z[1, 1] = missing
        Z[2, 2] = missing
        Z[3, 1] = missing
        for β in (0.8, 1.5), sw_every in (0, 1)
            marg, ES = exactposterior(Z, β, distributions, zeros(Int, 3, 3))
            r = simulatehiddenpotts(Z, β, distributions; nchains = n, num_iterations = 50, sw_every = sw_every, rng = Xoshiro(12), backend = BACKEND)
            @test agrees_independent(Array(r.Y), 3, marg, ES)
            @test agrees_missingdata(Array(r.Z), Z, distributions, marg)
        end
    end

    @testset "exact targets: hidden Potts, Beta + Dirac emissions with fixed pixels" begin
        distributions = [Beta(2.0, 5.0), Beta(5.0, 2.0), Dirac(0.0), Dirac(1.0)]
        Z = Matrix{Union{Missing, Float64}}(rand(Xoshiro(13), 3, 3))
        Z[1, 1] = 0.0
        Z[3, 3] = 1.0
        Z[2, 2] = missing
        Z[1, 3] = missing
        fixed = zeros(Int, 3, 3)
        fixed[1, 1] = 3
        fixed[3, 3] = 4
        marg, ES = exactposterior(Z, 1.0, distributions, fixed)
        for sw_every in (0, 1)
            r = simulatehiddenpotts(Z, 1.0, distributions; nchains = n, num_iterations = 50, sw_every = sw_every, rng = Xoshiro(14), backend = BACKEND)
            @test agrees_independent(Array(r.Y), 4, marg, ES)
            @test agrees_missingdata(Array(r.Z), Z, distributions, marg)
        end
    end

    @testset "exact targets: Potts model, conditional and unconditional" begin
        placeholder = [Normal(0.0, 1.0) for _ = 1:3]
        allmissing = Matrix{Union{Missing, Float64}}(missing, 3, 3)
        grid = Matrix{Union{Missing, Int}}([1 missing 2; missing missing 3; 1 missing missing])
        marg, ES = exactposterior(allmissing, 0.9, placeholder, coalesce.(grid, 0))
        for sw_every in (0, 1)
            Y = simulatepotts(grid, 3, 0.9; nchains = n, num_iterations = 50, sw_every = sw_every, rng = Xoshiro(15), backend = BACKEND)
            @test agrees_independent(Array(Y), 3, marg, ES)
        end
        marg, ES = exactposterior(allmissing, 1.1, placeholder, zeros(Int, 3, 3))
        for sw_every in (0, 1)
            Y = simulatepotts(3, 3, 3, 1.1; nchains = n, num_iterations = 50, sw_every = sw_every, rng = Xoshiro(16), backend = BACKEND)
            @test agrees_independent(Array(Y), 3, marg, ES)
        end
    end

    @testset "exact targets: more labels (counts in a UInt64, and by comparisons)" begin
        for q in (9, 17)
            placeholder = [Normal(0.0, 1.0) for _ = 1:q]
            marg, ES = exactposterior(Matrix{Union{Missing, Float64}}(missing, 2, 2), 1.2, placeholder, zeros(Int, 2, 2))
            for sw_every in (0, 1)
                Y = simulatepotts(2, 2, q, 1.2; nchains = n, num_iterations = 50, sw_every = sw_every, rng = Xoshiro(q), backend = BACKEND)
                @test agrees_independent(Array(Y), q, marg, ES)
            end
        end
    end

    @testset "agreement with the CPU sampler on a larger grid, above the critical value" begin
        # One data set of 120 × 120 pixels, with 30% missing values, and 10 chains (large clusters, whose labels are
        # joined by many threads at once)
        distributions = [Normal(-1.0, 0.7), Normal(0.0, 0.7), Normal(1.0, 0.7)]
        truth = simulatehiddenpotts(120, 120, 1.2, distributions; num_iterations = 200, rng = Xoshiro(21))
        Z = Matrix{Union{Missing, Float64}}(truth.Z[:, :, 1, 1])
        Z[rand(Xoshiro(22), 120, 120) .< 0.3] .= missing
        Ycpu = simulatehiddenpotts(Z, 1.2, distributions; nchains = 10, num_iterations = 40, rng = Xoshiro(23)).Y
        Ydev = Array(simulatehiddenpotts(Z, 1.2, distributions; nchains = 10, num_iterations = 40, rng = Xoshiro(23), backend = BACKEND).Y)
        accuracy(Y) = [mean(view(Y, :, :, 1, c) .== truth.Y[:, :, 1, 1]) for c in axes(Y, 4)]
        @test abs(mean(likefraction(Ydev)) - mean(likefraction(Ycpu))) < 0.01
        @test abs(mean(accuracy(Ydev)) - mean(accuracy(Ycpu))) < 0.01
        # The Potts model: one chain of 384 × 384 pixels, and four chains of 64 × 64
        for (m, nchains) in ((384, 1), (64, 4))
            fcpu = mean(likefraction(simulatepotts(m, m, 3, 1.2; nchains = nchains, num_iterations = 60, rng = Xoshiro(24))))
            fdev = mean(likefraction(Array(simulatepotts(m, m, 3, 1.2; nchains = nchains, num_iterations = 60, rng = Xoshiro(24), backend = BACKEND))))
            @test abs(fdev - fcpu) < 0.02
        end
    end

    @testset "data drawn on the device" begin
        # One label (q = 1), every value missing: the data are draws from the distribution itself
        for d in (Normal(1.0, 2.0), LogNormal(0.2, 0.5), Uniform(-1.0, 3.0), Exponential(2.0), Gamma(0.4, 1.5), Gamma(3.0, 0.5),
            Beta(0.5, 0.8), Beta(2.0, 5.0), Normal(1.0f0, 2.0f0), Gamma(2.5f0, 1.0f0), Beta(0.3f0, 0.6f0), TDist(5.0))
            Z = Array(simulatehiddenpotts(20, 20, 0.5, [d]; nchains = 250, num_iterations = 1, sw_every = 0, rng = Xoshiro(17), backend = BACKEND).Z)
            z = Float64.(vec(Z))
            @test eltype(Z) == eltype(d)
            @test abs(mean(z) - mean(d)) <= 5 * std(d) / sqrt(length(z))
            @test abs(var(z) / var(d) - 1) <= 0.05
        end
        Z = Array(simulatehiddenpotts(10, 10, 0.5, [Dirac(0.3)]; nchains = 3, num_iterations = 1, backend = BACKEND).Z)
        @test all(==(0.3), Z)
        # A distribution with integer values: drawn on the CPU and copied back
        Z = Array(simulatehiddenpotts(20, 20, 0.5, [Poisson(3.0)]; nchains = 250, num_iterations = 1, rng = Xoshiro(18), backend = BACKEND).Z)
        @test eltype(Z) == Int && abs(mean(Z) - 3) < 0.05 && abs(var(Z) - 3) < 0.15
    end

    @testset "output, reproducibility and launch configuration" begin
        distributions = [Normal(-1.0f0, 0.5f0), Normal(0.0f0, 0.5f0), Normal(1.0f0, 0.5f0)]
        Z = Matrix{Union{Missing, Float32}}(randn(Xoshiro(19), Float32, 24, 20))
        Z[rand(Xoshiro(20), 24, 20) .< 0.3] .= missing
        sim(seed) = simulatehiddenpotts(Z, 1.1, distributions; nchains = 7, num_iterations = 20, save_at = [0, 10, 20], rng = Xoshiro(seed), backend = BACKEND)
        r = sim(1)
        @test size(r.Y) == size(r.Z) == (24, 20, 3, 7)
        @test eltype(r.Y) == Int && eltype(r.Z) == Float32
        @test get_backend(r.Y) isa typeof(BACKEND) && get_backend(r.Z) isa typeof(BACKEND)
        Y, Zs = Array(r.Y), Array(r.Z)
        @test all(in(1:3), Y)
        observed = findall(!ismissing, Z)
        @test all(Zs[I, s, c] == Z[I] for I in observed, s = 1:3, c = 1:7)
        @test all(!isnan, Zs)
        r2 = sim(1)
        @test Array(r2.Y) == Y && Array(r2.Z) == Zs
        @test Array(sim(2).Y) != Y
        EXT.WORKGROUPSIZE[] = 64
        r3 = try
            sim(1)
        finally
            EXT.WORKGROUPSIZE[] = 256
        end
        @test Array(r3.Y) == Y && Array(r3.Z) == Zs
        pairs = EXT.GIBBS_PAIRS[]
        EXT.GIBBS_PAIRS[] = !pairs # the other Gibbs kernel: one or two chains per thread
        r4 = try
            sim(1)
        finally
            EXT.GIBBS_PAIRS[] = pairs
        end
        @test Array(r4.Y) == Y && Array(r4.Z) == Zs
        strip = EXT.SW_STRIP[]
        EXT.SW_STRIP[] = 1 # each thread of the Swendsen–Wang accumulation takes one label, not several
        r5 = try
            sim(1)
        finally
            EXT.SW_STRIP[] = strip
        end
        @test Array(r5.Y) == Y && Array(r5.Z) == Zs
        # Several Gibbs sweeps in one launch give the same results as one launch per half-sweep, with and without
        # Swendsen–Wang sweeps in between (with sw_every = 5, runs of four and five sweeps are fused, and the last two
        # sweeps are not)
        for sw_every in (0, 5)
            simsw(fused) = (EXT.FUSED_GIBBS[] = fused; Array(simulatehiddenpotts(Z, 1.1, distributions; nchains = 7, num_iterations = 11, sw_every = sw_every, rng = Xoshiro(3), backend = BACKEND).Y))
            Yfused, Yunfused = try
                simsw(true), simsw(false)
            finally
                EXT.FUSED_GIBBS[] = true
            end
            @test Yfused == Yunfused
        end
        # The Potts model
        grid = Matrix{Union{Missing, Int}}(rand(Xoshiro(21), 1:4, 10, 12))
        grid[rand(Xoshiro(22), 10, 12) .< 0.5] .= missing
        Yp = simulatepotts(grid, 4, 0.8; nchains = 3, sw_every = 2, backend = BACKEND)
        @test size(Yp) == (10, 12, 1, 3) && eltype(Yp) == Int
        Yp = Array(Yp)
        @test all(Yp[I, 1, c] == grid[I] for I in findall(!ismissing, grid), c = 1:3)
        @test all(in(1:4), Yp)
    end

    @testset "persistent chains on the device" begin
        distributions = [Normal(-1.0f0, 0.5f0), Normal(0.0f0, 0.5f0), Normal(1.0f0, 0.5f0)]
        Z = Matrix{Union{Missing, Float32}}(randn(Xoshiro(23), Float32, 16, 18))
        Z[rand(Xoshiro(24), 16, 18) .< 0.3] .= missing
        lat = PottsLattice(Z, distributions)
        par = PottsParams(lat, 1.1, distributions)
        chains = PottsChains(lat, 6, BACKEND; rng = Xoshiro(25))
        @test !chains.initialized
        run!(chains, lat, par, 5)
        @test chains.initialized && chains.iter == 5
        run!(chains, lat, PottsParams(lat, 0.6, distributions), 3; sw_every = 2) # new parameters: the chains continue
        @test chains.iter == 8
        Yout = allocate(BACKEND, Int, (16, 18, 6))
        @test getlabels!(Yout, chains, lat) === Yout
        @test all(in(1:3), Array(Yout))
        Zout = allocate(BACKEND, Float32, (16, 18, 6))
        copyto!(Zout, repeat(coalesce.(Z, NaN32), 1, 1, 6))
        drawZ!(Zout, chains, lat, par)
        Zh = Array(Zout)
        @test all(!isnan, Zh)
        @test all(Zh[I, c] == Z[I] for I in findall(!ismissing, Z), c = 1:6)
        initialize!(chains, lat, par)
        @test chains.iter == 0
        # Input validation
        @test_throws ArgumentError run!(chains, PottsLattice(Z, distributions), par, 1) # a different lattice object
        @test_throws DimensionMismatch getlabels!(allocate(BACKEND, Int, (16, 18, 5)), chains, lat)
        @test_throws DimensionMismatch drawZ!(allocate(BACKEND, Float32, (16, 17, 6)), chains, lat, par)
        latp = PottsLattice(16, 18)
        parp = PottsParams(latp, 1.0, 3)
        chainsp = run!(PottsChains(latp, 2, BACKEND), latp, parp, 1)
        @test_throws ArgumentError drawZ!(allocate(BACKEND, Float32, (16, 18, 2)), chainsp, latp, parp) # no distributions
        @test_throws ArgumentError PottsChains(lat, 0, BACKEND)
        @test_throws ArgumentError simulatepotts(4, 4, 33, 0.5; backend = BACKEND) # more than 32 labels
        @test_throws ArgumentError simulatepotts(4, 4, 3, 0.5; backend = :gpu)     # not a KernelAbstractions backend
        @test_throws ArgumentError PottsChains(lat, 2, :gpu)
    end

    @testset "parameters per chain" begin
        # With values per chain that are all equal, the kernels that index them (one or two chains per thread, and
        # several sweeps per launch) give the same results as with one value for all chains
        dA = [Normal(-1.0f0, 0.5f0), Normal(0.0f0, 0.5f0), Normal(1.0f0, 0.5f0)]
        Z = Matrix{Union{Missing, Float32}}(randn(Xoshiro(40), Float32, 14, 12))
        Z[rand(Xoshiro(41), 14, 12) .< 0.3] .= missing
        kw = (num_iterations = 12, sw_every = 5, save_at = [0, 12], backend = BACKEND)
        shared = simulatehiddenpotts(Z, 1.1, dA; nchains = 5, rng = Xoshiro(1), kw...)
        sharedu = simulatehiddenpotts(14, 12, 1.1, dA; nchains = 5, rng = Xoshiro(2), kw...)
        sharedp = simulatepotts(14, 12, 4, 0.8; nchains = 6, rng = Xoshiro(3), kw...)
        pairs = EXT.GIBBS_PAIRS[]
        EXT.SHARED_WHEN_EQUAL[] = false
        try
            for gibbspairs in (true, false)
                EXT.GIBBS_PAIRS[] = gibbspairs
                r = simulatehiddenpotts(Z, fill(1.1, 5), dA; rng = Xoshiro(1), kw...)
                @test Array(r.Y) == Array(shared.Y) && Array(r.Z) == Array(shared.Z)
                r = simulatehiddenpotts(14, 12, fill(1.1, 5), fill(dA, 5); rng = Xoshiro(2), kw...)
                @test Array(r.Y) == Array(sharedu.Y) && Array(r.Z) == Array(sharedu.Z)
                @test Array(simulatepotts(14, 12, 4, fill(0.8, 6); rng = Xoshiro(3), kw...)) == Array(sharedp)
            end
        finally
            EXT.SHARED_WHEN_EQUAL[] = true
            EXT.GIBBS_PAIRS[] = pairs
        end
        # Exact targets of two groups of chains, alternating: β = 0.5 and the emission distributions dA, and β = 1.2
        # and dB (unconditional simulation, and conditional simulation with β per chain)
        odd, even = 1:2:n, 2:2:n
        βs = [isodd(c) ? 0.5 : 1.2 for c = 1:n]
        dB = [Normal(4.0f0, 0.3f0), Normal(5.0f0, 0.3f0), Normal(6.0f0, 0.3f0)]
        placeholder = [Normal(0.0, 1.0) for _ = 1:3]
        allmissing = Matrix{Union{Missing, Float64}}(missing, 3, 3)
        for sw_every in (0, 1)
            r = simulatehiddenpotts(3, 3, βs, [isodd(c) ? dA : dB for c = 1:n]; num_iterations = 50, sw_every = sw_every, rng = Xoshiro(42), backend = BACKEND)
            Y, Zs = Array(r.Y), Array(r.Z)
            for (group, β, d) in ((odd, 0.5, dA), (even, 1.2, dB))
                marg, ES = exactposterior(allmissing, β, placeholder, zeros(Int, 3, 3))
                @test agrees_independent(Y[:, :, :, group], 3, marg, ES)
                @test agrees_missingdata(Zs[:, :, :, group], allmissing, d, marg)
            end
        end
        Zc = Matrix{Union{Missing, Float64}}(randn(Xoshiro(43), 3, 3) .* 0.8)
        Zc[1, 1] = missing
        Zc[2, 3] = missing
        dC = [Normal(-1.0, 0.6), Normal(0.0, 0.6), Normal(1.0, 0.6)]
        Y = Array(simulatehiddenpotts(Zc, βs, dC; num_iterations = 50, rng = Xoshiro(44), backend = BACKEND).Y)
        for (group, β) in ((odd, 0.5), (even, 1.2))
            marg, ES = exactposterior(Zc, β, dC, zeros(Int, 3, 3))
            @test agrees_independent(Y[:, :, :, group], 3, marg, ES)
        end
        # Emission distributions whose type differs between chains for a label: the data are drawn on the CPU
        dG = [Gamma(2.0f0, 1.0f0), Normal(0.0f0, 0.5f0), Normal(1.0f0, 0.5f0)]
        r = simulatehiddenpotts(3, 3, 0.5, [isodd(c) ? dA : dG for c = 1:4000]; num_iterations = 20, rng = Xoshiro(45), backend = BACKEND)
        marg, _ = exactposterior(allmissing, 0.5, placeholder, zeros(Int, 3, 3))
        Zs = Array(r.Z)
        @test agrees_missingdata(Zs[:, :, :, 1:2:4000], allmissing, dA, marg) && agrees_missingdata(Zs[:, :, :, 2:2:4000], allmissing, dG, marg)
        # Emission distributions per chain in conditional simulation are not supported on a device
        @test_throws ArgumentError simulatehiddenpotts(Zc, 1.0, [isodd(c) ? dC : dA for c = 1:4]; backend = BACKEND)
        # Persistent chains with one PottsParams per chain
        lat = PottsLattice(16, 18)
        pars = [PottsParams(lat, β, d) for (β, d) in ((0.5, dA), (1.2, dB), (0.9, dA), (1.5, dB))]
        chains = PottsChains(lat, 4, BACKEND; rng = Xoshiro(46))
        run!(chains, lat, pars, 6)
        Zout = allocate(BACKEND, Float32, (16, 18, 4))
        drawZ!(Zout, chains, lat, pars)
        Zh = Array(Zout)
        @test all(z -> -4 < z < 4, Zh[:, :, 1]) && all(z -> 2 < z < 9, Zh[:, :, 2])
        @test_throws DimensionMismatch run!(chains, lat, pars[1:3], 1)
    end

    @testset "helpers" begin
        # Philox: known-answer tests of Random123
        @test HMRF._philox2x32(0x00000000, 0x00000000, 0x00000000) == (0xff1dae59, 0x6cd10df2)
        @test HMRF._philox2x32(0xffffffff, 0xffffffff, 0xffffffff) == (0x2c3f628b, 0xab4fd7ad)
        @test HMRF._philox4x32(0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000) == (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)
        @test HMRF._philox4x32(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff) == (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)
        # Division by a multiplication and a shift
        rng = Xoshiro(26)
        divisors = unique([1:2000; [2^p + o for p = 11:30 for o = -1:1]; rand(rng, 1:typemax(Int32), 2000); typemax(Int32)])
        @test all(d -> divisionexact(d, rng), divisors)
        # Fixed-point log-weights: exact up to the scale, no overflow even for very large weights, and impossible labels
        # recorded in the masks
        distributions = [Normal(0.0, 1e-3), Normal(1.0, 1e-3), Dirac(5.0)]
        Z = Matrix{Union{Missing, Float64}}([0.0 1.0 0.5; 5.0 missing 2.0])
        lat = PottsLattice(Z, distributions)
        par = PottsParams(lat, 1.0, distributions)
        LW, mask, s = HMRF._fixedpoint_logweights(par)
        nobs = length(lat.obs_lin)
        @test maximum(abs, LW) * nobs <= 2.0^62
        @test all(mask[2:end] .== 0b100) && mask[1] == 0
        finite = isfinite.(par.logW)
        @test LW[finite] == round.(Int64, ldexp.(par.logW[finite], s))
        @test all(LW[.!finite] .== 0)
    end
end
