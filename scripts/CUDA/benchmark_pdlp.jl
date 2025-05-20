using DelimitedFiles
using MadNLP
using MadQP
using QPSReader
using QuadraticModels

include(joinpath("..", "common.jl"))
include("cuda_wrapper.jl")
include("qp_gpu.jl")

function run_benchmark(src, probs)
    nprobs = length(probs)
    results = zeros(nprobs, 5)
    for (k, prob) in enumerate(probs)
        @info prob
        qpdat = try
            import_mps(joinpath(src, prob))
        catch e
            @warn "Failed to import $prob: $e"
            continue
        end

        # Instantiate QuadraticModel
        qp = QuadraticModel(qpdat)
        new_qp = presolve_qp(qp)
        scaled_qp, Dr, Dc = scale_qp(new_qp)

        # Transfer data to the GPU
        qp_gpu = transfer_to_gpu(scaled_qp)

        try
            solver = MadQP.MPCSolver(
                qp_gpu;
                max_iter=300,
                tol=1e-7,
                linear_solver=MadNLPGPU.CUDSSSolver,
                cudss_algorithm=MadNLP.LDL,
                print_level=MadNLP.INFO,
                max_ncorr=3,
                bound_push=1.0,
                scaling=true,
                step_rule=MadQP.AdaptiveStep(0.995),
                regularization=MadQP.FixedRegularization(1e-8, -1e-8),
                rethrow_error=true,
                richardson_max_iter=0,
                richardson_tol=Inf,
            )
            res = MadQP.solve!(solver)
            results[k, 1] = Int(res.status)
            results[k, 2] = res.iter
            results[k, 3] = res.objective
            results[k, 4] = res.counters.total_time
            results[k, 5] = res.counters.linear_solver_time
        catch ex
            results[k, 4] = -1
            continue
        end
    end
    return results
end

variant = "medium"
src = "/home/amontoison/Argonne/LP_instances/$variant-problem-instances"
mps_files = filter(x -> endswith(x, ".mps.gz"), readdir(src))
results = run_benchmark(src, mps_files)
writedlm("benchmark-$variant-pdlp-gpu.txt", [mps_files results])
