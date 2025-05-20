using LinearAlgebra
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
new_qp = presolve_qp(qp)
scaled_qp, Dr, Dc = scale_qp(new_qp)

# Transfer data to the GPU
qp_gpu = transfer_to_gpu(scaled_qp)

solver = MadQP.MPCSolver(
    qp_gpu;
    max_iter=100,
    tol=1e-7,
    linear_solver=MadNLPGPU.CUDSSSolver,
    cudss_algorithm=MadNLP.LDL,
    print_level=MadNLP.INFO,
    scaling=true,
    max_ncorr=0,
    step_rule=MadQP.AdaptiveStep(0.995),
    regularization=MadQP.FixedRegularization(1e-8, -1e-8),
    rethrow_error=true,
    richardson_max_iter=0,
    richardson_tol=Inf,
)

MadQP.solve!(solver)
