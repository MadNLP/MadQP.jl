using DelimitedFiles
using MadNLP
using MadQP
using MadNLPHSL
using QPSReader
using QuadraticModels
using NLPModels

function NLPModels.cons!(qp::QuadraticModel{T}, x::Vector{T}, c::Vector{T}) where T
    return NLPModels.cons_lin!(qp, x, c)
end

function run_benchmark(src, probs)
    nprobs = length(probs)
    results = zeros(nprobs, 5)
    for (k, prob) in enumerate(probs)
        @info prob
        try
            qpdat = readqps(joinpath(src, prob))
        catch
            continue
        end
        qpdat = readqps(joinpath(src, prob))
        qp = QuadraticModel(qpdat)

        try
            solver = MadQP.MPCSolver(
                qp;
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
        end
    end
    return results
end

src = fetch_mm()
sif_files = filter(x -> endswith(x, ".SIF"), readdir(src))
results = run_benchmark(src, sif_files)
writedlm("benchmark-mm.txt", [sif_files results])
