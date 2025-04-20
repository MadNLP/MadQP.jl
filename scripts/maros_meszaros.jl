using MadNLP
using MadQP
using MadNLPHSL
using QPSReader
using QuadraticModels
using NLPModels

# Load Maros-Meszaros problems
src = fetch_mm()
sif_files = filter(x -> endswith(x, ".SIF"), readdir(src))

# Number of SIF files
num_sif_problems = length(sif_files)

# Choose instance to solve
for (i, sif) in enumerate(sif_files)
    pb_name = sif[1:end-4]
    println("Problem $i / $(num_sif_problems) : ", pb_name)

    # Path of the SIF file
    path_sif = joinpath(src, sif)

    # Read the SIF file
    qpdat = readqps(path_sif)

    # Instantiate QuadraticModel
    qp = QuadraticModel(qpdat)

    # Instantiate MadNLP
    solver = MadNLP.MadNLPSolver(
        qp;
        max_iter=100,
        linear_solver=LDLSolver,
        print_level=MadNLP.INFO,
        nlp_scaling=false,
        tol=1e-8,
        bound_push=100.0,
    )

    # Solve (increase max_ncorr to active Gondzio's correction)
    res = MadQP.solve!(solver; max_ncorr=0)
end
