```@meta
CollapsedDocStrings = true
```

# API

## Simulation

```@docs
simulatehiddenpotts

simulatepotts
```

## Persistent-chain interface

These types and functions are public but not exported. Access them with qualified names (e.g., `HiddenMarkovRandomFields.run!`), or import them explicitly with `using HiddenMarkovRandomFields: PottsLattice, PottsParams, PottsChain, run!`. They support algorithms that need the same chains across many calls, for example to continue them under new parameter values in a Monte Carlo EM algorithm ([Wei and Tanner, 1990](https://doi.org/10.1080/01621459.1990.10474930)), such as the neural EM algorithm ([Sainsbury-Dale et al., 2025](https://arxiv.org/abs/2501.04330)). See [Persistent chains](@ref) for an example.

Each task keeps its own scratch memory, so separate chains can be updated in parallel. Each chain must be updated by only one task at a time.

```@docs
HiddenMarkovRandomFields.PottsLattice

HiddenMarkovRandomFields.PottsParams

HiddenMarkovRandomFields.PottsChain

HiddenMarkovRandomFields.initialize!

HiddenMarkovRandomFields.run!

HiddenMarkovRandomFields.getlabels!

HiddenMarkovRandomFields.drawZ!
```

## GPU

Simulation on a GPU needs a GPU package that loads [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), such as CUDA.jl. Pass its backend, e.g. `CUDABackend()`, as the keyword argument `backend` of [`simulatehiddenpotts`](@ref) and [`simulatepotts`](@ref). For persistent chains, use `PottsChains` with the functions above. See [GPU sampling](@ref).

```@docs
HiddenMarkovRandomFields.PottsChains
```

## Utilities

```@docs
HiddenMarkovRandomFields.βcrit
```

## Index

```@index
```
