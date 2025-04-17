
using MadNLP
using MadQP
using MadNLPHSL
using QPSReader
using QuadraticModels
using NLPModels

function NLPModels.cons!(qp::QuadraticModel{T}, x::Vector{T}, c::Vector{T}) where T
    return NLPModels.cons_lin!(qp, x, c)
end

# Load Netlib
src = fetch_netlib()
# Choose instance to solve
qpdat = readqps(joinpath(src, "MAROS-R7.SIF"))
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
res= MadQP.solve!(solver; max_ncorr=0)
