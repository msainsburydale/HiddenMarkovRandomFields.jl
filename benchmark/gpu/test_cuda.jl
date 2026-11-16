# Runs the tests of sampling on a device (test/gpu.jl) on an NVIDIA GPU, with CUDA.jl. From the top level of the
# repository (the first command is needed only once):
#   julia --project=benchmark/gpu -e 'using Pkg; Pkg.instantiate()'
#   julia --project=benchmark/gpu --threads=8 benchmark/gpu/test_cuda.jl

using CUDA
CUDA.functional() || error("CUDA.jl cannot use a GPU on this machine")
println("GPU: ", CUDA.name(CUDA.device()))

const GPU_TEST_BACKEND = CUDABackend()
include(joinpath(@__DIR__, "..", "..", "test", "gpu.jl"))
