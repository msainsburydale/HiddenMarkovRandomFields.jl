# Run a subset of the test files with, e.g., Pkg.test(test_args = ["behaviour"])
const TESTFILES = ["exact", "behaviour", "gpu", "aqua"]

for name in ARGS
    name in TESTFILES || error("unknown test file \"$name\"; choose from $(join(TESTFILES, ", "))")
end
for name in TESTFILES
    (isempty(ARGS) || name in ARGS) && include("$name.jl")
end
