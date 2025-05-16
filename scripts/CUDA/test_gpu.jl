
using LinearAlgebra
using JuMP
using MadNLP
using MadQP

include(joinpath("..", "common.jl"))
include("cuda_wrapper.jl")
include("qp_gpu.jl")

src = fetch_netlib()
sif_files = filter(x -> endswith(x, ".SIF"), readdir(src))
num_sif_problems = length(sif_files)

sif = "WOODW.SIF"
path_sif = joinpath(src, sif)

# Read the SIF file
qpdat = import_mps(path_sif)

# Instantiate QuadraticModel
qp = QuadraticModel(qpdat)
# Transfer data to the GPU
qp_gpu = transfer_to_gpu(qp)

solver = MadQP.MPCSolver(
    qp_gpu;
    max_iter=100,
    tol=1e-7,
    linear_solver=MadNLPGPU.CUDSSSolver,
    cholmod_algorithm=MadNLP.LDL,
    print_level=MadNLP.INFO,
    scaling=true,
    max_ncorr=0,
    step_rule=MadQP.AdaptiveStep(0.995),
    regularization=MadQP.FixedRegularization(1e-8, -1e-8),
    rethrow_error=true,
)

MadQP.solve!(solver)

