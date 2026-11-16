# Contributing to HiddenMarkovRandomFields

Thank you for your interest in contributing to **HiddenMarkovRandomFields**! We welcome contributions of all sizes, from fixing typos to implementing new models. This document describes the code structure, the development workflow, and how to add a new model. A list of planned improvements is in [TODO.md](TODO.md).

## Code structure

```
src/
├── HiddenMarkovRandomFields.jl  # Module entry point: imports, exports, `public` names, includes
├── utility.jl                   # Helpers shared by the models (threading over chains, argument checks)
├── potts.jl                     # The (hidden) Potts model: data layout, sampler, and user-facing functions
└── gpu.jl                       # GPU sampling: PottsChains, and code shared with the extension
ext/
└── HiddenMarkovRandomFieldsKernelAbstractionsExt.jl  # GPU kernels (loaded with KernelAbstractions and Atomix)
test/
├── runtests.jl                  # Runs the test files below (or a subset; see "Development workflow")
├── enumeration.jl               # Exact targets computed by enumeration on tiny grids (used by exact.jl and gpu.jl)
├── exact.jl                     # Sampler output vs. exact targets
├── behaviour.jl                 # Output format, reproducibility, edge cases, input validation
├── gpu.jl                       # GPU sampling, run on KernelAbstractions' CPU backend (or on a GPU; see below)
└── aqua.jl                      # Automated quality checks (Aqua.jl)
docs/                            # Documenter.jl + DocumenterVitepress.jl (see docs/README.md)
benchmark/                       # Benchmark suite, run on pull requests by AirspeedVelocity.jl, and the tables
└── gpu/                         # Environment with CUDA.jl, for the GPU tests and the GPU columns of the tables
```

### The Potts sampler (`src/potts.jl`)

Two layers make up the sampler:

- **The user-facing functions** `simulatehiddenpotts` and `simulatepotts` are exported. They run independent chains in parallel and return their states after the iterations in `save_at`.
- **The persistent-chain interface** is public but not exported. It separates the quantities by how often they change:
  - `PottsLattice`: the data layout, built once per data set;
  - `PottsParams`: the parameter-dependent quantities, rebuilt when the parameters change;
  - `PottsChain`: the state of one chain;
  - `initialize!`, `run!`, `getlabels!` and `drawZ!`, which operate on chains.

The MCMC kernels (`gibbs_sweep!` and `sw_sweep!`) are internal. Each task keeps its scratch memory in a `PottsWorkspace`, rather than in the chains, so that many chains need little memory.

**Threads.** Threads are used at two levels:
- **Over chains.** `_simulate` distributes the chains over tasks with `_tforeach` (`src/utility.jl`).
- **Within a chain.** When there are fewer chains than tasks, the sweeps of each chain run in parallel over *blocks* of consecutive columns (`PottsLattice(...; nblocks)`).
  - A Gibbs half-sweep updates the blocks independently.
  - A Swendsen–Wang sweep runs in three phases: union-find within blocks in parallel; a sequential merge of the clusters that cross block boundaries; then the new labels, in parallel.

  `_run_parallel!` (`src/potts.jl`) starts its tasks once per call (`_withworkers`, `src/utility.jl`) rather than once per phase. In each phase, the tasks take blocks from an atomic counter (`_phase!`) and then wait at a spin barrier (`_SpinBarrier`). The last task to arrive at the barrier runs any sequential step (the merge) and resets the counter. A waiting task spins for a while and then yields, so the code also works with more tasks than threads. All the tasks are spawned, and the calling task waits for them: a sticky task, such as the main task, is resumed whenever it yields, so as a worker it could keep the other workers off its thread (this hung the tests on Julia 1.11). A task that fails aborts the barrier, so the others are released and the error is rethrown. Scratch arrays are sized before the tasks start, because nothing may be resized during a parallel phase.

Each block has its own random-number generator in the chain, and all floating-point accumulation happens in a fixed order. So the output must not depend on the number of tasks: `test/behaviour.jl` checks this, and `test/exact.jl` runs the exact-target tests with one block per column.

**Changing the kernels.** Changes should leave the exact-target tests passing, and should not make results depend on the number of tasks. For a pure refactor, the output for a given random-number generator should also be unchanged.

### GPU sampling (`src/gpu.jl` and `ext/`)

The GPU code is a package extension, loaded when KernelAbstractions and Atomix are (e.g., by `using CUDA`):
- **`src/gpu.jl`** holds what needs neither package:
  - the `PottsChains` type, with fallbacks that raise an informative error when the extension is not loaded;
  - the Philox random-number generators, and division by multiplication;
  - the host-side layouts of the likelihood weights;
  - the samplers of the emission distributions that run on the device.
- **The extension** holds the kernels, the device state of a `PottsChains`, and the methods of `initialize!`, `run!`, `getlabels!`, `drawZ!` and `_simulate_device`.

The labels of all chains are stored pixel by pixel, with the chains of each pixel next to each other. The comments at the top of the extension describe the layout, the random numbers and the four kernels of a Swendsen–Wang sweep.

**Results must not depend on how the work is divided among GPU threads.** Every random number comes from a counter-based generator whose counter identifies the pixel, the chain and the launch. Cluster sums are accumulated in fixed point with integer atomics, whose order does not matter. Union-find roots are always the smallest index of their cluster. `test/gpu.jl` checks this by changing the workgroup size, the Gibbs kernel (one or two chains per thread, and several sweeps per launch) and the number of labels each thread accumulates.

**Parameters per chain.** On the CPU, each chain simply gets its own `PottsParams`. On a GPU, the kernel arguments `E` (the table exp(βn)) and `pbond` are either scalars, as when every chain has the same β, or device arrays with one entry per chain. `_chainE` and `_chainpbond` return a scalar unchanged, so the kernels for parameters shared by all chains compile to the same code as before. When drawing data, each label's sampler is either shared by the chains or held in a device vector with one sampler per chain.

**Kernels must be type-stable.** The CPU backend runs kernels with dynamic dispatch, but a GPU cannot compile them. In particular, a closure (e.g., one passed to `ntuple`) must not capture a variable that is assigned more than once, since Julia boxes such variables. The tests on the CPU backend do not catch this, so run the CUDA tests after changing a kernel.

`test/gpu.jl` runs on KernelAbstractions' CPU backend as part of the test suite. To run the same tests on an NVIDIA GPU:
```bash
julia --project=benchmark/gpu -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/gpu --threads=8 benchmark/gpu/test_cuda.jl
```

**How the GPU code was tuned** (on an NVIDIA RTX A4500):
- **Gibbs sweeps.** Storing the chains of each pixel together, laying out each colour's weights in update order, 32-bit arithmetic, division by multiplication, and neighbour counts packed into one integer made the first version 1.1–2.0× faster. Updating two chains per thread, which then share the pixel's weights and one call of the random-number generator, made many chains another 1.3× faster. One large chain is limited by the memory bandwidth: at 2048 × 2048 pixels, a sweep moves about 84 MB at 560 GB/s, close to the GPU's 640 GB/s.
- **Small batches.** For runs of Gibbs sweeps of a few chains of up to 64 × 64 pixels, several sweeps share one launch, with each chain's labels in on-chip memory: up to 1.8× faster for 30 chains of 64 × 64, and 5× for one chain of 32 × 32. For more chains or larger grids it is slower (the labels are loaded and stored with a stride, and chains do not share random numbers), so it is not used there.
- **Swendsen–Wang sweeps.** Each thread of the accumulation combines four consecutive labels of its chain before adding them atomically to their root, which reduces contention when one cluster covers most of the grid; four was the best length, or close to it, for one large chain and for many small ones. Plain loads and stores in the union-find (a stale parent is still an ancestor), and finding the roots during the accumulation, made a sweep up to 1.6× faster. Storing the sums of each label in a plane of its own spreads the atomics of a large cluster over the memory: interleaving them was 30% slower above the critical value.
- **Parameters per chain.** Each thread of a Gibbs sweep loads the table exp(βn) of its chain with one 16-byte load, rather than five 4-byte loads, which cut the extra cost of parameters per chain from about 14% to 4% for many chains (storing the five values of all chains so that neighbouring threads read neighbouring values did not help).
- **Atomics.** With Atomix 1.2 and CUDA.jl 6.4, atomic operations ignore the requested memory ordering and compile to acquire-release ones; with Atomix 1.7 and CUDA.jl 6.5 they are relaxed, which made Swendsen–Wang sweeps of many chains up to 20% faster.
- **Reactant.jl.** A version with whole-array operations compiled by Reactant into one XLA program was 7–8× slower than the KernelAbstractions kernel for many chains, and took 32 s to compile its first program.

## Development workflow

1. **Fork and clone** the repository.
2. **Instantiate the package environment** from the repository root:
   ```bash
   julia --project=. -e "using Pkg; Pkg.instantiate()"
   ```
3. **Make your changes.** It is often easiest to prototype interactively first: load the package, `import` the internal functions you need, and redefine methods on the fly. Move the definitions into the appropriate source file once they work.
4. **Export or declare `public`** any new user-facing names in `src/HiddenMarkovRandomFields.jl`, and give each one a docstring.
5. **Write or update the documentation** (see [docs/README.md](docs/README.md)). Every exported or public name must appear in the API page.
6. **Write or update the tests** in `test/`, then run them with several threads:
   ```bash
   julia --project=. --threads=4 -e "using Pkg; Pkg.test()"
   ```
   To run a subset of the test files, pass their names as test arguments:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.test(test_args = ["behaviour"])'
   ```
7. **Format the code** with [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl), using the settings in `.JuliaFormatter.toml`: run `using JuliaFormatter; format(".")` from the repository root. A scheduled workflow otherwise opens a pull request with the formatting changes.
8. **Open a pull request** describing your changes. Pull requests are tested on Linux, macOS and Windows, and benchmarked against the main branch.

Please follow the [Julia style guide](https://docs.julialang.org/en/v1/manual/style-guide/) where possible.

## Adding a new model

The package name anticipates other hidden Markov random fields: for example, other neighbourhood structures, irregular graphs, or other models for the labels. To add a model:

1. **Create a source file**, `src/<model>.jl`, and `include` it in `src/HiddenMarkovRandomFields.jl`. Reuse the helpers in `src/utility.jl`: `_tforeach` runs chains in parallel with one workspace per task, and `_check_save_at` validates the iterations to save.
2. **Follow the conventions of the Potts functions**, so that the models can be used interchangeably:
   - a `simulate<model>` function with methods for unconditional simulation (grid dimensions) and conditional simulation (incomplete data, with `missing` marking the values to simulate);
   - the keyword arguments `nchains`, `num_iterations`, `save_at`, `rng` and `ntasks`;
   - output arrays of dimension `nrow × ncol × length(save_at) × nchains`;
   - chains seeded from `rng`, so that the results do not depend on the number of threads.
3. **Test the sampler against exact targets.** On grids small enough to enumerate every configuration, compare the sampler's output with the exact distribution, as in `test/exact.jl`. Also test the behaviour of the user-facing functions, as in `test/behaviour.jl`.
4. **Document the model** with a section in `docs/src/model.md`, an example in `docs/src/examples.md`, and its functions in `docs/src/API.md`.
5. **Add benchmarks** for the new sampler to `benchmark/benchmarks.jl`.
