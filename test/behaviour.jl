# Behaviour of the user-facing functions and the persistent-chain API: output format, reproducibility, fixed pixels,
# edge cases and input validation

using HiddenMarkovRandomFields, Test, Random
using HiddenMarkovRandomFields: PottsLattice, PottsParams, PottsChain, initialize!, run!, getlabels!, drawZ!
using Distributions: Normal, Beta, Dirac

@testset "behaviour" begin
    distributions = [Normal(-1.0f0, 0.2f0), Normal(0.0f0, 0.2f0), Normal(1.0f0, 0.2f0)]

    @testset "fewer than q labels observed (regression)" begin
        # every observed value is near μ₁, so only label 1 is ever observed; an old initialiser inferred q = 1 here
        Z = Matrix{Union{Missing, Float32}}(fill(-1.0f0, 12, 12) .+ 0.05f0 .* randn(Xoshiro(3), Float32, 12, 12))
        Z[4:9, 4:9] .= missing
        r = simulatehiddenpotts(Z, 0.3, distributions; nchains = 50, num_iterations = 20, rng = Xoshiro(4))
        @test all(1 .<= r.Y .<= 3)
        @test Set(r.Y[4:9, 4:9, 1, :]) == Set(1:3) # all labels are sampled in the hole
        @test !any(isnan, r.Z)
    end

    @testset "reproducible, and independent of the number of tasks" begin
        Z = Matrix{Union{Missing, Float32}}(randn(Xoshiro(5), Float32, 20, 20))
        Z[rand(Xoshiro(6), 20, 20) .< 0.3] .= missing
        r1 = simulatehiddenpotts(Z, 1.1, distributions; nchains = 9, num_iterations = 30, rng = Xoshiro(7), ntasks = 1)
        r2 = simulatehiddenpotts(Z, 1.1, distributions; nchains = 9, num_iterations = 30, rng = Xoshiro(7), ntasks = 4)
        r3 = simulatehiddenpotts(Z, 1.1, distributions; nchains = 9, num_iterations = 30, rng = Xoshiro(8), ntasks = 4)
        @test r1.Y == r2.Y && r1.Z == r2.Z
        @test r1.Z != r3.Z
        Y1 = simulatepotts(20, 20, 4, 1.2; nchains = 9, num_iterations = 30, rng = Xoshiro(7), ntasks = 1)
        Y2 = simulatepotts(20, 20, 4, 1.2; nchains = 9, num_iterations = 30, rng = Xoshiro(7), ntasks = 4)
        @test Y1 == Y2
    end

    @testset "output format" begin
        Z = Matrix{Union{Missing, Float32}}(randn(Xoshiro(9), Float32, 16, 12))
        Z[1:5, 1:5] .= missing
        r = simulatehiddenpotts(Z, 0.8, distributions; nchains = 3, num_iterations = 10, save_at = [1, 5, 10])
        @test size(r.Y) == (16, 12, 3, 3) && size(r.Z) == (16, 12, 3, 3)
        @test eltype(r.Y) == Int && eltype(r.Z) == Float32
        obs = .!ismissing.(Z)
        @test all(r.Z[:, :, s, c][obs] == Z[obs] for s = 1:3, c = 1:3)
        m = simulatehiddenpotts(16, 12, 0.8, distributions; nchains = 2, num_iterations = 10)
        @test size(m.Y) == size(m.Z) == (16, 12, 1, 2)
        @test eltype(m.Z) == Float32
        Y = simulatepotts(10, 8, 3, 1.2; nchains = 4, num_iterations = 10, save_at = [0, 10])
        @test size(Y) == (10, 8, 2, 4) && eltype(Y) == Int
        @test all(in(1:3), Y)
    end

    @testset "save_at = 0 returns the initial state" begin
        Z = Matrix{Union{Missing, Float32}}(randn(Xoshiro(10), Float32, 10, 10))
        Z[3:6, 3:6] .= missing
        r0 = simulatehiddenpotts(Z, 0.8, distributions; nchains = 2, num_iterations = 0, rng = Xoshiro(11))
        r = simulatehiddenpotts(Z, 0.8, distributions; nchains = 2, num_iterations = 10, save_at = [0, 10], rng = Xoshiro(11))
        @test r0.Y[:, :, 1, :] == r.Y[:, :, 1, :]
        @test r.Y[:, :, 1, :] != r.Y[:, :, 2, :]
    end

    @testset "complete and all-missing data" begin
        Zc = randn(Xoshiro(12), Float32, 8, 6)
        r = simulatehiddenpotts(Zc, 0.8, distributions; nchains = 2, num_iterations = 5)
        @test all(r.Z[:, :, 1, c] == Zc for c = 1:2)
        @test eltype(r.Z) == Float32
        r = simulatehiddenpotts(Matrix{Missing}(missing, 8, 6), 0.8, distributions; nchains = 2, num_iterations = 5)
        @test size(r.Z) == (8, 6, 1, 2) && eltype(r.Z) == Float32 && !any(isnan, r.Z)
        Yc = rand(Xoshiro(13), 1:3, 8, 6)
        @test simulatepotts(Yc, 3, 1.0; nchains = 2, num_iterations = 5) == cat(Yc, Yc; dims = 4)
        Y = simulatepotts(Matrix{Missing}(missing, 8, 6), 3, 1.0; num_iterations = 5)
        @test size(Y) == (8, 6, 1, 1) && all(in(1:3), Y)
    end

    @testset "Potts: conditional simulation keeps the observed labels" begin
        Y = simulatepotts(20, 16, 4, 1.0; num_iterations = 20, rng = Xoshiro(14))[:, :, 1, 1]
        Y₁ = Matrix{Union{Missing, Int}}(Y)
        miss = rand(Xoshiro(15), 20, 16) .< 0.4
        Y₁[miss] .= missing
        R = simulatepotts(Y₁, 4, 1.0; nchains = 5, num_iterations = 10, save_at = [0, 10])
        @test size(R) == (20, 16, 2, 5)
        @test all(R[:, :, s, c][.!miss] == Y[.!miss] for s = 1:2, c = 1:5)
        @test all(in(1:4), R)
        # q exceeds the number of distinct observed labels
        Y₂ = Matrix{Union{Missing, Int}}(fill(1, 6, 6))
        Y₂[2:5, 2:5] .= missing
        R = simulatepotts(Y₂, 3, 0.2; nchains = 50, num_iterations = 5)
        @test Set(R[2:5, 2:5, 1, :]) == Set(1:3)
    end

    @testset "observed values equal to Dirac atoms are fixed" begin
        d = [Beta(2.0, 5.0), Beta(5.0, 2.0), Dirac(0.0), Dirac(1.0)]
        Z = Matrix{Union{Missing, Float64}}(rand(Xoshiro(16), 10, 10))
        Z[1:2, :] .= 0.0
        Z[9:10, :] .= 1.0
        Z[4:7, 4:7] .= missing
        r = simulatehiddenpotts(Z, 1.0, d; nchains = 4, num_iterations = 10, save_at = [0, 10])
        @test all(==(3), r.Y[1:2, :, :, :]) && all(==(4), r.Y[9:10, :, :, :])
        @test all(in(1:2), r.Y[3, :, :, :]) && all(in(1:2), r.Y[8, :, :, :]) # values in (0, 1): Beta labels only
        @test all(r.Z[:, :, s, c][.!ismissing.(Z)] == Z[.!ismissing.(Z)] for s = 1:2, c = 1:4)
        # a missing value drawn from a Dirac label equals its atom
        hole = r.Z[4:7, 4:7, :, :]
        labels = r.Y[4:7, 4:7, :, :]
        @test all(hole[labels .== 3] .== 0.0) && all(hole[labels .== 4] .== 1.0)
    end

    @testset "persistent chains" begin
        Z = Matrix{Union{Missing, Float32}}(randn(Xoshiro(17), Float32, 12, 10))
        Z[3:8, 2:6] .= missing
        lat = PottsLattice(Z, distributions)
        par = PottsParams(lat, 0.8, distributions)

        # run! initialises a new chain; a chain can continue under new parameters
        chain = PottsChain(lat, Xoshiro(1))
        run!(chain, lat, par, 10)
        @test chain.iter == 10
        par2 = PottsParams(lat, 1.2, [Normal(-1.1f0, 0.3f0), Normal(0.1f0, 0.3f0), Normal(1.1f0, 0.3f0)])
        run!(chain, lat, par2, 5; sw_every = 2)
        @test chain.iter == 15
        Y = getlabels!(zeros(Int, size(Z)), chain, lat)
        @test all(in(1:3), Y)
        Zc = drawZ!(coalesce.(Z, 0.0f0), chain, lat, par2)
        @test Zc[.!ismissing.(Z)] == Z[.!ismissing.(Z)]

        # initialize! resets the chain; running in pieces equals running in one go
        a = PottsChain(lat, Xoshiro(2))
        b = PottsChain(lat, Xoshiro(2))
        initialize!(a, lat, par)
        run!(a, lat, par, 10)
        run!(b, lat, par, 4)
        run!(b, lat, par, 6)
        @test getlabels!(zeros(Int, size(Z)), a, lat) == getlabels!(zeros(Int, size(Z)), b, lat)
        initialize!(a, lat, par)
        @test a.iter == 0

        # chains advanced concurrently in separate tasks (each with its own workspace) match sequential runs
        seq = [run!(PottsChain(lat, Xoshiro(c)), lat, par, 20) for c = 1:8]
        par_chains = [PottsChain(lat, Xoshiro(c)) for c = 1:8]
        @sync for c in par_chains
            Threads.@spawn run!(c, lat, par, 20)
        end
        @test all(getlabels!(zeros(Int, size(Z)), seq[c], lat) == getlabels!(zeros(Int, size(Z)), par_chains[c], lat) for c = 1:8)

        # mismatched arguments
        other = PottsLattice(4, 4)
        @test_throws DimensionMismatch run!(chain, other, PottsParams(other, 0.5, 3), 1)
        @test_throws DimensionMismatch run!(chain, lat, PottsParams(other, 0.5, distributions), 1)
        @test_throws DimensionMismatch getlabels!(zeros(Int, 3, 3), chain, lat)
        @test_throws DimensionMismatch drawZ!(zeros(Float32, 3, 3), chain, lat, par)
        pottschain = PottsChain(other, Xoshiro(3))
        @test_throws ArgumentError drawZ!(zeros(4, 4), pottschain, other, PottsParams(other, 0.5, 3))
        @test_throws DimensionMismatch run!(PottsChain(PottsLattice(4, 4; nblocks = 2), Xoshiro(4)), other, PottsParams(other, 0.5, 3), 1)
    end

    @testset "parallelism within a chain: results do not depend on the number of tasks" begin
        d = [Normal(-1.0, 0.5), Normal(0.0, 0.5), Normal(1.0, 0.5), Dirac(2.0)]
        Z = Matrix{Union{Missing, Float64}}(randn(Xoshiro(20), 40, 37))
        Z[rand(Xoshiro(21), 40, 37) .< 0.1] .= 2.0 # fixed pixels (Dirac atom)
        Z[5:30, 10:25] .= missing
        lat = PottsLattice(Z, d; nblocks = 10)
        par = PottsParams(lat, 1.2, d)
        results = map((1, 2, 4)) do ntasks
            chain = PottsChain(lat, Xoshiro(22))
            run!(chain, lat, par, 30; ntasks = ntasks)
            run!(chain, lat, par, 7; sw_every = 3, ntasks = ntasks)
            (getlabels!(zeros(Int, size(Z)), chain, lat), drawZ!(coalesce.(Z, 0.0), chain, lat, par))
        end
        @test results[1] == results[2] == results[3]
        @test all(results[1][1][Z .=== 2.0] .== 4) # fixed pixels keep their label

        # simulatehiddenpotts on a grid large enough for parallelism within chains: one chain (all tasks within the
        # chain) and three chains (tasks both over and within chains)
        z = simulatehiddenpotts(130, 140, 1.0, d[1:3]; num_iterations = 20, rng = Xoshiro(23)).Z[:, :, 1, 1]
        Z₁ = Matrix{Union{Missing, Float64}}(z)
        Z₁[rand(Xoshiro(24), 130, 140) .< 0.3] .= missing
        bigger = PottsLattice(Z₁, d[1:3])
        @test HiddenMarkovRandomFields._splittasks(bigger, 4, 1) == (1, 4)
        @test HiddenMarkovRandomFields._splittasks(bigger, 6, 3) == (3, 2)
        @test HiddenMarkovRandomFields._splittasks(PottsLattice(40, 40), 4, 1) == (1, 1) # too small to parallelise
        for nchains in (1, 3)
            r1 = simulatehiddenpotts(Z₁, 1.0, d[1:3]; nchains = nchains, num_iterations = 10, rng = Xoshiro(25), ntasks = 1)
            r2 = simulatehiddenpotts(Z₁, 1.0, d[1:3]; nchains = nchains, num_iterations = 10, rng = Xoshiro(25), ntasks = 6)
            @test r1.Y == r2.Y && r1.Z == r2.Z
        end
    end

    @testset "spin barrier and worker tasks" begin
        H = HiddenMarkovRandomFields
        # a task passes a barrier only after every task has arrived, and the sequential step runs once per barrier
        arrivals = Threads.Atomic{Int}(0)
        violations = Threads.Atomic{Int}(0)
        steps = Ref(0)
        H._withworkers(4) do _, barrier
            for phase = 1:1000
                Threads.atomic_add!(arrivals, 1)
                H._wait!(barrier) do
                    arrivals[] == 4phase || Threads.atomic_add!(violations, 1)
                    steps[] += 1
                end
                arrivals[] >= 4phase || Threads.atomic_add!(violations, 1)
            end
        end
        @test steps[] == 1000 && violations[] == 0 && arrivals[] == 4000

        # more tasks than threads
        n = 2 * Threads.nthreads() + 1
        total = Threads.Atomic{Int}(0)
        H._withworkers(n) do _, barrier
            for _ = 1:50
                Threads.atomic_add!(total, 1)
                H._wait!(() -> nothing, barrier)
            end
        end
        @test total[] == 50n

        # a task that fails, or a sequential step that fails, releases the other tasks and its error is rethrown
        @test_throws ErrorException H._withworkers(4) do w, barrier
            w == 3 && error("task 3 failed")
            H._wait!(() -> nothing, barrier)
        end
        @test_throws ErrorException H._withworkers(3) do _, barrier
            H._wait!(() -> error("sequential step failed"), barrier)
        end
    end

    @testset "parameters per chain" begin
        d = [Normal(-1.0, 0.5), Normal(0.0, 0.5), Normal(1.0, 0.5)]
        Z = Matrix{Union{Missing, Float64}}(randn(Xoshiro(30), 12, 10))
        Z[rand(Xoshiro(31), 12, 10) .< 0.3] .= missing
        grid = Matrix{Union{Missing, Int}}(rand(Xoshiro(32), 1:3, 12, 10))
        grid[rand(Xoshiro(33), 12, 10) .< 0.5] .= missing
        # Values per chain that are all equal give the same results as one value for all chains
        kw = (num_iterations = 10, save_at = [0, 10])
        @test simulatehiddenpotts(Z, fill(0.9, 4), d; rng = Xoshiro(1), kw...) == simulatehiddenpotts(Z, 0.9, d; nchains = 4, rng = Xoshiro(1), kw...)
        @test simulatehiddenpotts(Z, 0.9, fill(d, 4); rng = Xoshiro(2), kw...) == simulatehiddenpotts(Z, 0.9, d; nchains = 4, rng = Xoshiro(2), kw...)
        @test simulatehiddenpotts(12, 10, fill(0.9, 4), fill(d, 4); rng = Xoshiro(3), kw...) == simulatehiddenpotts(12, 10, 0.9, d; nchains = 4, rng = Xoshiro(3), kw...)
        @test simulatepotts(12, 10, 3, fill(1.1, 3); rng = Xoshiro(4), kw...) == simulatepotts(12, 10, 3, 1.1; nchains = 3, rng = Xoshiro(4), kw...)
        @test simulatepotts(grid, 3, fill(1.1, 3); rng = Xoshiro(5), kw...) == simulatepotts(grid, 3, 1.1; nchains = 3, rng = Xoshiro(5), kw...)
        # nchains defaults to the number of values given, and chain c has β[c] and distributions[c]
        likefraction(Y) = (count(Y[1:(end - 1), :] .== Y[2:end, :]) + count(Y[:, 1:(end - 1)] .== Y[:, 2:end])) / (2 * 12 * 10 - 12 - 10)
        Y = simulatepotts(12, 10, 3, [0.0, 3.0]; num_iterations = 30, rng = Xoshiro(6))
        @test size(Y) == (12, 10, 1, 2)
        @test likefraction(Y[:, :, 1, 1]) < 0.6 && likefraction(Y[:, :, 1, 2]) > 0.9
        far = [Normal(100.0, 0.1), Normal(101.0, 0.1), Normal(102.0, 0.1)]
        r = simulatehiddenpotts(12, 10, 0.5, [d, far]; num_iterations = 5, rng = Xoshiro(7))
        @test size(r.Z) == (12, 10, 1, 2)
        @test all(<(10), r.Z[:, :, 1, 1]) && all(>(90), r.Z[:, :, 1, 2])
        @test eltype(simulatehiddenpotts(4, 4, 0.5, [[Normal(0.0f0, 1.0f0)], [Normal(0.0, 1.0)]]; num_iterations = 1).Z) == Float64
        # Errors: numbers of chains that disagree, different numbers of labels, and Dirac atoms that differ between
        # chains in conditional simulation (they are allowed in unconditional simulation)
        @test_throws ArgumentError simulatehiddenpotts(Z, [0.1, 0.2], d; nchains = 3)
        @test_throws ArgumentError simulatehiddenpotts(Z, 0.5, [d, d]; nchains = 3)
        @test_throws ArgumentError simulatehiddenpotts(12, 10, [0.1, 0.2], [d, d, d])
        @test_throws ArgumentError simulatehiddenpotts(12, 10, 0.5, [d, d[1:2]])
        @test_throws ArgumentError simulatehiddenpotts(Z, Float64[], d)
        @test_throws ArgumentError simulatepotts(4, 4, 3, [0.5, -0.1])
        zeroinflated = Matrix{Union{Missing, Float64}}([0.0 0.5; missing 0.2])
        @test_throws ArgumentError simulatehiddenpotts(zeroinflated, 0.5, [[Beta(2.0, 2.0), Dirac(0.0)], [Beta(2.0, 2.0), Dirac(1.0)]])
        @test size(simulatehiddenpotts(4, 4, 0.5, [[Beta(2.0, 2.0), Dirac(0.0)], [Beta(2.0, 2.0), Dirac(1.0)]]; num_iterations = 1).Z) == (4, 4, 1, 2)
    end

    @testset "input validation" begin
        d = [Normal(0.0, 1.0), Normal(1.0, 1.0)]
        Z = Matrix{Union{Missing, Float64}}(randn(Xoshiro(18), 5, 5))
        Z[1] = missing
        @test_throws ArgumentError simulatehiddenpotts(Z, -0.1, d)
        @test_throws ArgumentError simulatehiddenpotts(Z, Inf, d)
        @test_throws ArgumentError simulatehiddenpotts(Z, [0.1, 0.2], d; nchains = 3)
        @test_throws ArgumentError simulatehiddenpotts(Z, 0.5, d; nchains = 0)
        @test_throws ArgumentError simulatehiddenpotts(Z, 0.5, d; num_iterations = -1)
        @test_throws ArgumentError simulatehiddenpotts(Z, 0.5, d; num_iterations = 5, save_at = 6)
        @test_throws ArgumentError simulatehiddenpotts(Z, 0.5, d; num_iterations = 5, save_at = Int[])
        @test_throws ArgumentError simulatehiddenpotts(Z, 0.5, d; sw_every = -1)
        @test_throws ArgumentError simulatehiddenpotts([0.5 0.5], 0.5, [Dirac(0.0), Dirac(1.0)]) # zero likelihood
        @test_throws ArgumentError simulatepotts(5, 5, 0, 0.5)
        @test_throws ArgumentError simulatepotts(5, 5, 256, 0.5)
        Y₁ = Matrix{Union{Missing, Int}}(fill(1, 4, 4))
        Y₁[2] = missing
        Y₁[1] = 4
        @test_throws ArgumentError simulatepotts(Y₁, 3, 0.5)
        Y₁[1] = 0
        @test_throws ArgumentError simulatepotts(Y₁, 3, 0.5)
        @test_throws DimensionMismatch PottsLattice(falses(3, 3), zeros(Int, 3, 2), zeros(3, 3))
        @test_throws ArgumentError PottsLattice(4, 4; nblocks = 0)
        @test_throws ArgumentError PottsLattice(4, 4; nblocks = 5)
        @test PottsLattice(4, 4; nblocks = 4).nblocks == 4
        lat = PottsLattice(4, 4)
        @test_throws ArgumentError run!(PottsChain(lat, Xoshiro(1)), lat, PottsParams(lat, 0.5, 3), 1; ntasks = 0)
        @test_throws ArgumentError simulatepotts(4, 4, 3, 0.5; ntasks = 0)
        @test size(simulatepotts(4, 4, 2, [0.5]; num_iterations = 1)) == (4, 4, 1, 1) # β in a one-element container
    end

    @testset "βcrit" begin
        @test HiddenMarkovRandomFields.βcrit(2) ≈ log(1 + sqrt(2))
    end
end
