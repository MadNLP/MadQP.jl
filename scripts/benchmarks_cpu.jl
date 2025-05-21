using DelimitedFiles
using MadNLP
using MadQP
using MadNLPHSL
using QPSReader
using QuadraticModels

include("common.jl")

function run_benchmark(src, probs; reformulate::Bool=false)
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
        qp = QuadraticModel(qpdat)
        new_qp = presolve_qp(qp)
        scaled_qp = scale_qp(new_qp)
        qp_cpu = reformulate ? standard_form_qp(scaled_qp) : scaled_qp

        try
            solver = MadQP.MPCSolver(
                qp_cpu;
                max_iter=300,
                linear_solver=Ma27Solver,
                print_level=MadNLP.INFO,
                max_ncorr=3,
                bound_push=1.0,
            )
            res = MadQP.solve!(solver)
            results[k, 1] = Int(res.status)
            results[k, 2] = res.iter
            results[k, 3] = res.objective
            results[k, 4] = res.counters.total_time
            results[k, 5] = res.counters.linear_solver_time
        catch ex
            results[k, 4] = -1
            @warn "Failed to solve $prob: $ex"
            continue
        end
    end
    return results
end

src = fetch_netlib()
name_results = "benchmark-netlib.txt"

# src = fetch_mm()
# name_results = "benchmark-mm.txt"

reformulate = true
sif_files = filter(x -> endswith(x, ".SIF"), readdir(src))
results = run_benchmark(src, sif_files; reformulate)
writedlm(name_results, [sif_files results])
