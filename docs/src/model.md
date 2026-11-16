# Model and algorithms

## The hidden Potts model

Consider a regular grid of $n_r \times n_c$ pixels. Each pixel $i$ has a label $Y_i \in \{1, \dots, q\}$. The labels $\boldsymbol{Y}$ follow a $q$-state Potts model with inverse-temperature parameter $\beta \geq 0$:

```math
p(\boldsymbol{Y} \mid \beta) \propto \exp\Big(\beta \sum_{i \sim j} \mathbb{1}(Y_i = Y_j)\Big),
```

where the sum runs over all pairs of first-order (4-nearest) neighbours, and pixels on the edge of the grid simply have fewer neighbours (free boundary). Neighbouring pixels tend to share a label, more strongly as $\beta$ increases. On the infinite lattice, the model has a phase transition at the critical value $\beta_c = \log(1 + \sqrt{q})$, available as [`βcrit`](@ref HiddenMarkovRandomFields.βcrit). Below $\beta_c$, realisations are disordered. Above it, realisations are dominated by large clusters of a single label.

In the hidden Potts model, the labels are not observed. Instead, each pixel has an observation $Z_i$ that depends only on its label:

```math
Z_i \mid Y_i = k \;\sim\; F_k \quad \text{independently}, \qquad k = 1, \dots, q,
```

where the *emission distributions* $F_1, \dots, F_q$ are any univariate distributions from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl). The number of labels is the number of emission distributions. In the Potts model itself, the labels are observed directly.

## Unconditional and conditional simulation

[`simulatehiddenpotts`](@ref) and [`simulatepotts`](@ref) offer two kinds of simulation:

- **Unconditional simulation** draws the labels from the Potts model, and then the data from the emission distributions.
- **Conditional simulation** starts from incomplete data, with observed values $\boldsymbol{Z}_1$ and missing values $\boldsymbol{Z}_2$. It draws the labels from $p(\boldsymbol{Y} \mid \boldsymbol{Z}_1)$, and then the missing values from the emission distributions given their labels. For the Potts model, the known labels are held fixed and the others are simulated.

Each pixel is therefore one of three kinds:

| Kind | Updated by the sampler? | Contribution to the conditional distribution of its label |
|:--|:--|:--|
| missing | yes | none |
| observed | yes | the likelihood weight $w_{ik} = f_k(z_i)$ |
| fixed | no | its label is known |

A pixel is *fixed* when its label is known. In the Potts model, these are the observed labels. In the hidden Potts model, they are observed values equal to the atom $a$ of a point-mass emission distribution, `Dirac(a)`. Under continuous emissions, an observation equal to $a$ has probability zero, so its label is known with certainty. This allows, for example, zero-inflated or bounded data (as in the sea-ice example in [Examples](@ref)) to be modelled with Beta emissions for the interior of $(0, 1)$ and `Dirac(0.0)` and `Dirac(1.0)` for the boundary values.

## Gibbs sampling

Given the labels of its neighbours, the label of pixel $i$ has the conditional distribution

```math
P(Y_i = k \mid \boldsymbol{Y}_{-i}, \boldsymbol{Z}_1) \propto w_{ik} \exp(\beta n_{ik}),
```

where $n_{ik}$ is the number of neighbours of pixel $i$ with label $k$, and $w_{ik} = 1$ if pixel $i$ is missing. Colour the grid as a chequerboard. Then the pixels of one colour are conditionally independent given the pixels of the other colour, so each half of the grid is updated in a single pass. One Gibbs sweep updates both halves in turn.

The sampler precomputes the likelihood weights once per parameter value. It also tabulates $\exp(\beta n)$ for the five possible neighbour counts $n = 0, \dots, 4$, so a pixel update involves no special functions or allocations.

## Swendsen–Wang updates

Gibbs sampling changes one pixel at a time. Near and above $\beta_c$, large clusters of one label rarely change, and Gibbs sampling mixes slowly. The Swendsen–Wang algorithm [(Swendsen and Wang, 1987)](https://doi.org/10.1103/PhysRevLett.58.86) changes whole clusters at once, using the Edwards–Sokal representation of the model [(Edwards and Sokal, 1988)](https://doi.org/10.1103/PhysRevD.38.2009). Each sweep has two steps:

1. Every pair of neighbours with equal labels is joined by a bond independently with probability $1 - e^{-\beta}$. The bonds partition the grid into clusters.
2. Each cluster $C$ draws a new label, with probability proportional to the product of the likelihood weights of its pixels, $\prod_{i \in C} w_{ik}$. A cluster that contains a fixed pixel keeps that pixel's label.

When there are data, the likelihood weights make it hard for large clusters to change label. This is the usual difficulty of Swendsen–Wang with an external field [(Higdon, 1998)](https://doi.org/10.1080/01621459.1998.10473712), and one reason the two updates are combined. Each MCMC iteration is a chequerboard Gibbs sweep, preceded every `sw_every` iterations by a Swendsen–Wang sweep. The default, `sw_every = 1`, runs both in every iteration, and `sw_every = 0` gives Gibbs sampling only.

## Initialisation

The chains start close to a typical state, to shorten burn-in:

- Fixed pixels take their known labels.
- Observed pixels take their maximum-likelihood labels.
- Missing pixels are filled outward from the assigned pixels by a breadth-first flood fill, after 1% of them are seeded at random. Below $\beta_c$, each pixel is drawn from the Potts conditional distribution given its assigned neighbours. Above $\beta_c$, each pixel takes the most common label among its assigned neighbours.

With no data at all (unconditional simulation), the labels start uniformly at random below $\beta_c$. Above $\beta_c$, they start as random blocks with a few random perturbations.

## Multiple chains, threads and reproducibility

The keyword argument `nchains` sets the number of independent chains, and `ntasks` (by default, the number of Julia threads) the number of tasks that run in parallel. The chains are distributed over the tasks. When there are fewer chains than tasks, the remaining tasks share the work of each chain, for grids of roughly 64 × 64 pixels or more; see [Performance](@ref).

To share the work of a chain, the grid is divided into blocks of consecutive columns:
- **Gibbs sweeps.** The blocks are updated in parallel, since the pixels of one chequerboard colour are conditionally independent.
- **Swendsen–Wang sweeps.** Clusters are first found within each block in parallel. The clusters that cross block boundaries are then joined, and finally every cluster draws its new label in parallel.

Each block has its own random-number generator, seeded from `rng`, so the results depend only on `rng`, not on the number of tasks or threads.

Each chain can have its own parameters: `β` can be a vector with one value per chain, and the emission distributions a vector with one vector of distributions per chain. One call then simulates, for example, the training data of a neural estimator, with each field drawn from its own parameter values; see [Examples](@ref).

The keyword argument `save_at` selects the iterations whose states are returned, which allows, for example, burn-in to be assessed. Iteration 0 is the initial state.

To keep chains between calls, use the persistent-chain interface described in [API](@ref). This is useful, for example, to continue the chains under new parameter values in a Monte Carlo EM algorithm ([Wei and Tanner, 1990](https://doi.org/10.1080/01621459.1990.10474930)), such as the neural EM algorithm ([Sainsbury-Dale et al., 2025](https://arxiv.org/abs/2501.04330)).

## GPU sampling

Set the keyword argument `backend` to a [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) backend, such as `CUDABackend()` after `using CUDA`. Then [`simulatehiddenpotts`](@ref) and [`simulatepotts`](@ref) run all `nchains` chains together on the GPU, and return arrays on the GPU. Loading a GPU package loads KernelAbstractions, which loads the package extension that holds the GPU code. For persistent chains on a GPU, use [`PottsChains`](@ref HiddenMarkovRandomFields.PottsChains).

The GPU runs the same algorithms as the CPU:
- **Initialisation.** The chains are initialised on the CPU, as described above, and then copied to the GPU.
- **Gibbs sweeps.** One kernel per chequerboard colour updates every pixel of that colour in every chain, with each GPU thread updating two chains. The labels are stored pixel by pixel, with the chains of each pixel next to each other, so that neighbouring GPU threads read and write neighbouring bytes. For a few chains on a small grid, consecutive sweeps run in one kernel, which keeps each chain's labels in on-chip memory.
- **Swendsen–Wang sweeps.** The bonds of all chains are opened in parallel, and their clusters are found by union-find with atomic operations. Each cluster then sums the log-likelihood weights of its pixels, and draws its new label from these sums. The sums are accumulated in fixed point with integer atomics, so they do not depend on the order of the additions. Each GPU thread first adds up the weights of a few consecutive pixels of the same cluster, which reduces contention when one cluster covers most of the grid.
- **Data.** The missing values are drawn on the GPU for `Normal`, `LogNormal`, `Uniform`, `Exponential`, `Gamma`, `Beta` and `Dirac` emissions. For other distributions, they are drawn on the CPU and copied back.

Random numbers come from the counter-based generators Philox2x32-10 and Philox4x32-10 [(Salmon et al., 2011)](https://doi.org/10.1145/2063384.2063405), indexed by the pixel, the chain and the step of the algorithm. Results are therefore reproducible given `rng`, and don't depend on how the work is divided among GPU threads. They differ from those of the CPU sampler, though.

The GPU code is written with KernelAbstractions and tested on NVIDIA GPUs with CUDA.jl. Other GPUs with 64-bit atomic operations, e.g. AMD GPUs with AMDGPU.jl, may also work. Apple GPUs (Metal.jl) won't, since they lack 64-bit atomics. At most 32 labels are supported. See [Performance](@ref) for timings.

## References

- Edwards, R. G. and Sokal, A. D. (1988). Generalization of the Fortuin–Kasteleyn–Swendsen–Wang representation and Monte Carlo algorithm. *Physical Review D*, 38, 2009–2012. [doi:10.1103/PhysRevD.38.2009](https://doi.org/10.1103/PhysRevD.38.2009)
- Higdon, D. M. (1998). Auxiliary variable methods for Markov chain Monte Carlo with applications. *Journal of the American Statistical Association*, 93, 585–595. [doi:10.1080/01621459.1998.10473712](https://doi.org/10.1080/01621459.1998.10473712)
- Sainsbury-Dale, M., Zammit-Mangion, A., Cressie, N. and Huser, R. (2025). Neural parameter estimation with incomplete data. [arXiv:2501.04330](https://arxiv.org/abs/2501.04330)
- Salmon, J. K., Moraes, M. A., Dror, R. O. and Shaw, D. E. (2011). Parallel random numbers: as easy as 1, 2, 3. *Proceedings of the 2011 International Conference for High Performance Computing, Networking, Storage and Analysis*, 16:1–16:12. [doi:10.1145/2063384.2063405](https://doi.org/10.1145/2063384.2063405)
- Swendsen, R. H. and Wang, J.-S. (1987). Nonuniversal critical dynamics in Monte Carlo simulations. *Physical Review Letters*, 58, 86–88. [doi:10.1103/PhysRevLett.58.86](https://doi.org/10.1103/PhysRevLett.58.86)
- Wei, G. C. G. and Tanner, M. A. (1990). A Monte Carlo implementation of the EM algorithm and the poor man's data augmentation algorithms. *Journal of the American Statistical Association*, 85, 699–704. [doi:10.1080/01621459.1990.10474930](https://doi.org/10.1080/01621459.1990.10474930)
