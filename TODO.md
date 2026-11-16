# TODO List

A checklist of planned tasks, improvements, and ideas for the package. Feel free to update this file as tasks are completed, added, or changed. 

- **GPU: emission distributions per chain in conditional simulation.** On a GPU, chains with their own emission distributions are supported in unconditional simulation only, where the distributions affect only the data drawn. In conditional simulation, they would need likelihood weights per chain in the Gibbs kernels and in the Swendsen–Wang accumulation and singleton draws. (On the CPU, they are supported.)
- **GPU batches of several data sets.** A `PottsChains` holds the chains of one data set, so a few chains of a small grid (e.g., 30 chains of 64 × 64) make a small batch, whose time is set by the overhead of the kernel launches. For 32 such data sets, as in batched Monte Carlo EM, the GPU is only 1.6–1.9× faster than 32 CPU threads, against 3.4× for about as many chains in one batch (1024 chains of 64 × 64). Batches that hold several lattices (each with its own missing and fixed pixels, weights and β) would close that gap.
- **Release**
  - Create the GitHub repository `msainsburydale/HiddenMarkovRandomFields.jl`. Add the `CODECOV_TOKEN` and `DOCUMENTER_KEY` secrets, and enable GitHub Pages from the `gh-pages` branch.
  - Register the package in the General registry (`@JuliaRegistrator register`).
- **Downstream packages**
  - **IncompleteData.** Replace `src/HiddenPotts/Simulation.jl` with `using HiddenMarkovRandomFields`. Keep `HiddenPottsEMSimulator` in that repository, built on the public persistent-chain API (`PottsLattice`, `PottsParams`, `PottsChain`, `initialize!`, `run!`, `drawZ!`).
  - **NeuralEstimators.jl.** It also exports `simulatepotts`, which uses the old algorithm, so loading both packages causes a name clash. Remove it there, or depend on this package and re-export this one.