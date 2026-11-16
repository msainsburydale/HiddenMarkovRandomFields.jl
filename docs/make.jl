using Pkg
Pkg.activate(@__DIR__)
Pkg.resolve()      # HiddenMarkovRandomFields is taken from the parent directory ([sources] in docs/Project.toml)
Pkg.instantiate()
using Documenter, DocumenterVitepress, HiddenMarkovRandomFields

# Links to the source code on GitHub need a git commit to point to. Without one (e.g., before the first commit, or in a
# copy of the repository that is not a git repository), build the documentation without them.
hascommit = try
    success(pipeline(`git -C $(dirname(@__DIR__)) rev-parse --verify --quiet HEAD`; stdout = devnull, stderr = devnull))
catch # git not installed
    false
end
hascommit || @warn "No git commit found: building the documentation without links to the source code"
sourcelinks = hascommit ? (repo = Remotes.GitHub("msainsburydale", "HiddenMarkovRandomFields.jl"),) : (remotes = nothing,)

makedocs(;
    sitename = "HiddenMarkovRandomFields.jl",
    authors = "Matthew Sainsbury-Dale",
    sourcelinks...,
    modules = [HiddenMarkovRandomFields],
    checkdocs = :public, # every exported or `public` name must appear in the documentation
    pages = [
        "Home" => "index.md",
        "Model and algorithms" => "model.md",
        "Examples" => "examples.md",
        "Performance" => "performance.md",
        "API" => "API.md"
    ],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/msainsburydale/HiddenMarkovRandomFields.jl",
        devbranch = "main",
        devurl = "dev")
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/msainsburydale/HiddenMarkovRandomFields.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true
)
