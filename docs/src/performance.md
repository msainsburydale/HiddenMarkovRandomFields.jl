# Performance

The timings are wall times of one MCMC iteration in milliseconds (ms), excluding initialisation, for a hidden Potts model with $q = 3$ labels, Gaussian emissions and $\beta = 1.0$, close to the critical value. *Gibbs only* is a chequerboard Gibbs sweep (`sw_every = 0`), and *Gibbs + SW* adds a Swendsen–Wang sweep (`sw_every = 1`, the default). They were measured on an AMD Ryzen Threadripper PRO 5995WX CPU, on 1 thread and on 32 threads, and on an NVIDIA RTX A4500 GPU, with Julia 1.12.6 and CUDA.jl 6.5.1.

## Summary

- **GPU.** A GPU is the fastest for all but the smallest workloads, often by far: up to 17× faster than 32 threads. Only for 30 chains of 32 × 32 pixels with Swendsen–Wang are 32 threads faster, by about 10%.
- **Threads.** With many chains, 32 threads are up to 28× faster than 1 thread. A single chain uses several threads from about 64 × 64 pixels, and is then up to 16× faster on 32 threads.
- **Swendsen–Wang** makes an iteration 5–7× as expensive as a Gibbs sweep alone on 1 thread, and up to about 20× on 32 threads and 30× on a GPU, where Gibbs sweeps parallelise better. It pays off near and above the critical value, where Gibbs sampling alone mixes slowly (see [Burn-in and Swendsen–Wang updates](@ref)).
- **Unconditional simulation** costs about as much as conditional simulation with Gibbs sweeps only, and up to about 40% less with Swendsen–Wang, since there are no likelihood weights to add up over the clusters.
- **Parameters per chain** cost nothing extra on the CPU, and up to about 5% on a GPU.

## Time per iteration

```@example performance
using HiddenMarkovRandomFields, CairoMakie # hide
rows = split.(readlines(joinpath(pkgdir(HiddenMarkovRandomFields), "benchmark", "results.csv"))[2:end], ',') # hide
rows = filter(r -> r[1] == "time per iteration", rows) # hide
devices = unique(r[7] for r in rows) # hide
colours = ["#7F7F7F", "#4063D8", "#389826"] # hide
algorithms = ("Gibbs only", "Gibbs + SW") # hide
fig = Figure(size = (700, 500), fontsize = 15) # hide
axs = Axis[] # hide
for (i, algorithm) in enumerate(algorithms), (j, nchains) in enumerate(("1", "30", "1024")) # hide
    ax = Axis(fig[i, j]; title = i == 1 ? (nchains == "1" ? "1 chain" : "$nchains chains") : "", xscale = log2, yscale = log10, # hide
        xticks = ([64, 256, 1024], ["64²", "256²", "1024²"]), xminorticks = [32, 128, 512], xminorticksvisible = true, xminorgridvisible = true, # hide
        xlabel = "Grid size", yticks = (10.0 .^ (-3:3), ["0.001", "0.01", "0.1", "1", "10", "100", "1000"])) # hide
    for (device, colour) in zip(devices, colours), (simulation, linestyle) in (("conditional", :solid), ("unconditional", :dash)) # hide
        r = filter(r -> r[2] == simulation && r[5] == nchains && r[6] == algorithm && r[7] == device, rows) # hide
        scatterlines!(ax, parse.(Int, getindex.(r, 4)), parse.(Float64, getindex.(r, 8)); color = colour, linestyle, markersize = 6) # hide
    end # hide
    i == 1 && hidexdecorations!(ax; grid = false, ticks = false) # hide
    j > 1 && hideydecorations!(ax; grid = false, ticks = false) # hide
    push!(axs, ax) # hide
end # hide
linkaxes!(axs...) # hide
Label(fig[1:2, 0], "Time per iteration (ms)"; rotation = π / 2, tellheight = false) # hide
for (i, algorithm) in enumerate(algorithms) # hide
    Label(fig[i, 4], algorithm; rotation = -π / 2, font = :bold, tellheight = false) # hide
end # hide
Legend(fig[3, 0:4], [[LineElement(color = c) for c in colours[eachindex(devices)]]; [LineElement(color = :black, linestyle = s) for s in (:solid, :dash)]], # hide
    [devices; "conditional"; "unconditional"]; orientation = :horizontal, framevisible = false, tellheight = true, tellwidth = false) # hide
fig # hide
```

The table gives the time per iteration in ms on 1 thread / 32 threads / a GPU, with the fastest in bold, for three of the grid sizes in the figure.

| Grid | Chains | Simulation | Gibbs only | Gibbs + SW |
|:--|--:|:--|--:|--:|
| 64² | 1 | conditional | 0.019 / 0.0084 / **0.0039** | 0.096 / 0.057 / **0.026** |
|  |  | unconditional | 0.018 / 0.0098 / **0.0039** | 0.085 / 0.051 / **0.023** |
|  | 30 | conditional | 0.56 / 0.022 / **0.0040** | 2.9 / 0.11 / **0.048** |
|  |  | unconditional | 0.56 / 0.022 / **0.0040** | 2.6 / 0.098 / **0.045** |
|  | 1024 | conditional | 19 / 0.68 / **0.063** | 98 / 3.5 / **1.0** |
|  |  | unconditional | 19 / 0.67 / **0.064** | 87 / 3.1 / **0.87** |
| 256² | 1 | conditional | 0.29 / 0.033 / **0.0055** | 1.5 / 0.40 / **0.034** |
|  |  | unconditional | 0.29 / 0.032 / **0.0056** | 1.4 / 0.35 / **0.030** |
|  | 30 | conditional | 9.0 / 0.34 / **0.032** | 46 / 1.8 / **0.55** |
|  |  | unconditional | 8.9 / 0.34 / **0.032** | 41 / 1.7 / **0.49** |
|  | 1024 | conditional | 310 / 11 / **1.0** | 1600 / 58 / **24** |
|  |  | unconditional | 300 / 11 / **1.0** | 1400 / 50 / **16** |
| 1024² | 1 | conditional | 4.7 / 0.33 / **0.042** | 33 / 4.1 / **0.48** |
|  |  | unconditional | 4.6 / 0.28 / **0.042** | 26 / 3.1 / **0.29** |
|  | 30 | conditional | 140 / 8.5 / **0.51** | 980 / 120 / **8.4** |
|  |  | unconditional | 140 / 6.4 / **0.50** | 770 / 110 / **7.3** |

## Parameters per chain

Each chain can have its own β and emission distributions (see [Unconditional simulation](@ref)), at no extra cost on the CPU and for up to about 5% more time per iteration on a GPU. One call of [`simulatehiddenpotts`](@ref) that simulates a batch of training data, 1024 fields of 64 × 64 pixels with 200 iterations, each field with its own β (uniform on (0, 1.5)) and emission distributions, takes 0.62 seconds on 32 threads and 0.18 seconds on a GPU, including initialisation and drawing the data.

## Using a GPU

- **Keep the results on the GPU.** `simulatehiddenpotts` returns arrays on the GPU, and copying them to the CPU, e.g. with `Array(Y)`, can take longer than several iterations. Use them there, e.g. in a neural network, and continue the chains with [`PottsChains`](@ref HiddenMarkovRandomFields.PottsChains), whose [`drawZ!`](@ref HiddenMarkovRandomFields.drawZ!) draws the data on the GPU.
- **Memory.** Per pixel and chain, the labels take 1 byte, and the Swendsen–Wang sweeps another $14 + 8q$ bytes. The output of `simulatehiddenpotts` takes 8 bytes per pixel, chain and saved iteration for `Y`, plus the size of the data type for `Z`.

## Reproducing the results

[benchmark/table.jl](https://github.com/msainsburydale/HiddenMarkovRandomFields.jl/blob/main/benchmark/table.jl) measures the times, writes them to [benchmark/results.csv](https://github.com/msainsburydale/HiddenMarkovRandomFields.jl/blob/main/benchmark/results.csv), from which the figure is drawn, and prints the table. From the top level of the repository:

```bash
# On the CPU (the first command is needed only once)
julia --project=benchmark -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
julia --project=benchmark --threads=auto benchmark/table.jl
# With an NVIDIA GPU, in the environment benchmark/gpu, which has CUDA.jl
julia --project=benchmark/gpu -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/gpu --threads=auto benchmark/table.jl
```
