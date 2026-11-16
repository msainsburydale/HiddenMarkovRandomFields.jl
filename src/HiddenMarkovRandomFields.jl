module HiddenMarkovRandomFields

using Random
using Random: AbstractRNG, Xoshiro
using Distributions: Distributions, logpdf, sampler, Dirac

export simulatehiddenpotts, simulatepotts

# Persistent-chain API (documented and stable, but not exported: names such as run! clash too easily)
public βcrit, PottsLattice, PottsParams, PottsChain, PottsChains, initialize!, run!, getlabels!, drawZ!

include("utility.jl")
include("potts.jl")
include("gpu.jl") # sampling on a GPU; the kernels are in ext/HiddenMarkovRandomFieldsKernelAbstractionsExt.jl

end
