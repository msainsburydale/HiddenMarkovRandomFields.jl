# Generates docs/src/assets/sampling.gif (shown in the README and on the home page of the documentation): Gibbs and
# Swendsen–Wang sampling of a hidden Potts model with q = 3 labels for several values of β, starting from random labels.
#
# Run from the top level of the repository:  julia --project=docs docs/animation.jl

using HiddenMarkovRandomFields, CairoMakie, Random
using HiddenMarkovRandomFields: PottsLattice, PottsParams, PottsChain, initialize!, run!, getlabels!

const OUTPUT = joinpath(@__DIR__, "src", "assets", "sampling.gif")
CairoMakie.activate!(px_per_unit = 1) # one pixel per unit of figure size, to keep the file small

n = 72                   # the grid is n × n
q = 3                    # number of labels
μ = [-1.0, 0.0, 1.0]     # emission distributions: Normal(μ[k], σ) for label k
σ = 0.4
βs = [0.6, 0.9, 1.0, 1.3]
frames = 0:60            # iterations shown

# All permutations of 1:q
allperms(q) = q == 1 ? [[1]] : [insert!(copy(p), i, q) for p in allperms(q - 1) for i = 1:q]

# The labels Y, permuted to agree as much as possible with the labels of the previous frame. A Swendsen–Wang sweep gives
# each cluster a uniformly random new label, so without this, a large cluster would change colour in most frames; the
# labels of the Potts model are exchangeable, so the permuted labels are as likely as Y.
aligned(Y, previous, perms) = argmax(p -> count(i -> p[Y[i]] == previous[i], eachindex(Y)), perms)[Y]

# One chain per value of β, started from uniformly random labels (the initialisation for β = 0)
lattice = PottsLattice(n, n)
chains = [PottsChain(lattice, Xoshiro(k)) for k in eachindex(βs)]
foreach(chain -> initialize!(chain, lattice, PottsParams(lattice, 0.0, q)), chains)
params = [PottsParams(lattice, β, q) for β in βs]
perms = allperms(q)

# Data Z = μ[Y] + σε, with the same noise ε in every frame so that the data change only where the labels change
ε = randn(Xoshiro(0), n, n)
labels = [Observable(getlabels!(zeros(Int, n, n), chain, lattice)) for chain in chains]
data = [lift(Y -> μ[Y] .+ σ .* ε, Y) for Y in labels]
iteration = Observable(0)

colours = ["#CB3C33", "#389826", "#9558B2"] # Julia red, green, purple
fig = Figure(size = (660, 360), figure_padding = (4, 8, 4, 4))
Label(fig[0, 1:length(βs)], lift(i -> "Gibbs + Swendsen–Wang, iteration $i", iteration); fontsize = 15, font = :bold)
Label(fig[1, 0], "Labels"; rotation = π / 2, tellheight = false)
Label(fig[2, 0], "Data"; rotation = π / 2, tellheight = false)
for (k, β) in enumerate(βs)
    top = Axis(fig[1, k]; title = "β = $β", aspect = DataAspect(), yreversed = true)
    bottom = Axis(fig[2, k]; aspect = DataAspect(), yreversed = true)
    for ax in (top, bottom)
        hidedecorations!(ax)
        hidespines!(ax)
    end
    heatmap!(top, lift(permutedims, labels[k]); colormap = cgrad(colours, q; categorical = true), colorrange = (0.5, q + 0.5))
    heatmap!(bottom, lift(permutedims, data[k]); colormap = :viridis, colorrange = (-2, 2))
end
rowgap!(fig.layout, 4)
colgap!(fig.layout, 6)

record(fig, OUTPUT, frames; framerate = 8) do i
    for k in eachindex(chains)
        run!(chains[k], lattice, params[k], i - iteration[]; sw_every = 1) # Gibbs and Swendsen–Wang, up to iteration i
        labels[k][] = aligned(getlabels!(zeros(Int, n, n), chains[k], lattice), labels[k][], perms)
    end
    iteration[] = i
end
println("Wrote $OUTPUT ($(round(filesize(OUTPUT) / 2^20; digits = 2)) MiB)")
