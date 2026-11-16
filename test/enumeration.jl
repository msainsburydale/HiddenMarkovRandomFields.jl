# Exact computations by enumeration on small grids, and their comparison with the output of independent chains
# (shared by exact.jl and gpu.jl)

using Distributions: logpdf, var
using Statistics: mean, std

# Number of like-neighbour pairs (first-order neighbourhood, free boundary)
function likepairs(Y::AbstractMatrix)
    nrow, ncol = size(Y)
    s = 0
    for j = 1:ncol, i = 1:nrow
        i < nrow && (s += Y[i, j] == Y[i + 1, j])
        j < ncol && (s += Y[i, j] == Y[i, j + 1])
    end
    return s
end

# Exact posterior of the labels given the data: Z has `missing` at missing pixels; fixedlabel > 0 marks fixed pixels.
# Observed (non-fixed) pixels contribute logpdf(distributions[k], z). Returns the label marginals (nrow × ncol × q) and E[likepairs].
function exactposterior(Z::AbstractMatrix, β, distributions, fixedlabel::AbstractMatrix{<:Integer})
    nrow, ncol = size(Z)
    n = nrow * ncol
    q = length(distributions)
    Y = zeros(Int, nrow, ncol)
    logp = Float64[]
    marg = zeros(nrow, ncol, q)
    S = Float64[]
    configs = Vector{Matrix{Int}}()
    for code = 0:(q ^ n - 1)
        c = code
        for i = 1:n
            Y[i] = c % q + 1
            c ÷= q
        end
        ok = all(fixedlabel[i] == 0 || Y[i] == fixedlabel[i] for i = 1:n)
        ok || continue
        lp = β * likepairs(Y)
        for i = 1:n
            (ismissing(Z[i]) || fixedlabel[i] > 0) && continue
            lp += logpdf(distributions[Y[i]], Z[i])
        end
        push!(logp, lp)
        push!(S, likepairs(Y))
        push!(configs, copy(Y))
    end
    w = exp.(logp .- maximum(logp))
    w ./= sum(w)
    for (wt, Yc) in zip(w, configs), i = 1:n
        marg[i + (Yc[i] - 1) * n] += wt
    end
    return marg, sum(w .* S)
end

# Empirical label marginals (nrow × ncol × q) of the final states of the chains in Y (nrow × ncol × nsave × nchains)
function empiricalmarginals(Y::AbstractArray{<:Integer, 4}, q)
    nrow, ncol, _, n = size(Y)
    P = zeros(nrow, ncol, q)
    for c = 1:n, j = 1:ncol, i = 1:nrow
        P[i, j, Y[i, j, end, c]] += 1
    end
    return P ./ n
end

# Agreement of the empirical marginals and mean number of like-neighbour pairs of the final states of independent
# chains (Y: nrow × ncol × nsave × nchains) with their exact values
function agrees_independent(Y, q, marg, ES; k = 4.5)
    n = size(Y, 4)
    P = empiricalmarginals(Y, q)
    se = sqrt.(max.(marg .* (1 .- marg), 0) ./ n) # max: marginals of fixed pixels are 1 up to rounding
    okP = all(abs.(P .- marg) .<= k .* se .+ 1e-3)
    S = [likepairs(view(Y, :, :, size(Y, 3), c)) for c = 1:n]
    okS = abs(mean(S) - ES) <= k * std(S) / sqrt(n) + 1e-3
    (okP && okS) || @warn "Disagreement" P marg mean(S) ES
    return okP && okS
end

# The mean and second moment of the values drawn at each missing pixel of Z agree with those of the exact mixture
# (Zs: nrow × ncol × nsave × nchains; marg: the exact label marginals)
function agrees_missingdata(Zs, Z, distributions, marg; k = 4.5)
    ok = true
    for I in CartesianIndices(Z)
        ismissing(Z[I]) || continue
        p = marg[I, :]
        m1 = sum(p[l] * mean(distributions[l]) for l in eachindex(p))
        m2 = sum(p[l] * (var(distributions[l]) + mean(distributions[l])^2) for l in eachindex(p))
        z = Float64.(vec(Zs[I, size(Zs, 3), :]))
        n = length(z)
        ok &= abs(mean(z) - m1) <= k * std(z) / sqrt(n) + 1e-3
        ok &= abs(mean(z .^ 2) - m2) <= k * std(z .^ 2) / sqrt(n) + 1e-3
    end
    ok || @warn "Disagreement of the missing data"
    return ok
end
