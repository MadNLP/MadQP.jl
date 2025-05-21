using LinearAlgebra
using MadNLP
using MadQP

include("common.jl")
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
scaled_qp = scale_qp(new_qp)
standard_qp = standard_form_qp(scaled_qp)

# Transfer data to the GPU
for operator in (false, true)

    qp_gpu = transfer_to_gpu(standard_qp; operator)

    for (kkt, algo) in (# (MadQP.NormalKKTSystem, MadNLP.CHOLESKY),
                        (MadNLP.SparseKKTSystem, MadNLP.LDL),)

        solver = MadQP.MPCSolver(
            qp_gpu;
            max_iter=100,
            tol=1e-7,
            kkt_system=kkt,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=algo,
            print_level=MadNLP.INFO,
            scaling=true,
            max_ncorr=0,
            step_rule=MadQP.AdaptiveStep(0.995),
            regularization=MadQP.FixedRegularization(1e-8, -1e-8),
            rethrow_error=true,
        )

        MadQP.solve!(solver)
    end
end
