# Instructions for contributing to the documentation

The [package documentation](https://msainsburydale.github.io/HiddenMarkovRandomFields.jl/dev/) is built with [Documenter.jl](https://documenter.juliadocs.org/stable/) and [DocumenterVitepress.jl](https://luxdl.github.io/DocumenterVitepress.jl/dev/). Its sources are the files in [docs/src](src) and the docstrings in [src](../src).

The figures on the examples page are generated when the documentation is built, by the `@example` blocks in [docs/src/examples.md](src/examples.md), so there is no separate step for regenerating them. So is the figure on the performance page, from the timings in [benchmark/results.csv](../benchmark/results.csv), which [benchmark/table.jl](../benchmark/table.jl) writes together with the tables. The animation on the home page and in the README is made by [docs/animation.jl](animation.jl): run `julia --project=docs docs/animation.jl` from the top level of the repository to regenerate it.

## Building the documentation locally

From the top level of the repository:

```bash
julia --project=docs docs/make.jl
```

The package is taken from the parent directory (see `[sources]` in [docs/Project.toml](Project.toml)), so the build reflects local changes to the code. The first build installs the Node.js packages listed in [docs/package.json](package.json), which takes a minute or so. To preview the built site:

```bash
julia -e 'using LiveServer; serve(dir = "docs/build/1")'
```

(`LiveServer` is not a documentation dependency: install it once in your global environment with `julia -e 'using Pkg; Pkg.add("LiveServer")'`.)

## Conventions

- Every exported name, and every name declared `public` in `src/HiddenMarkovRandomFields.jl`, must appear in an `@docs` block (the build enforces this with `checkdocs = :public`).
- Pushes to `main` build and deploy the documentation automatically (`.github/workflows/Documentation.yml`); pull requests get a preview.
