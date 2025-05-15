
using LinearAlgebra
using JuMP
using MadQP
using MadNLP

using ExaModels

include("cuda_wrapper.jl")

# Build QP using JuMP
A = rand(2, 10)
model = Model()
@variable(model,  0 <= x[1:10])
# @constraint(model, A * x .== 0.0)
@constraint(model, sum(x) == 1.0)
@objective(model, Min, -ones(10)' * x)

# Pass it to the GPU using ExaModels
qp = ExaModels.ExaModel(model; backend=CUDABackend())

solver = MadQP.MPCSolver(
    qp;
    max_iter=100,
    linear_solver=MadNLPGPU.CUDSSSolver,
    cholmod_algorithm=MadNLP.LDL,
    print_level=MadNLP.INFO,
    scaling=true,
    tol=1e-8,
    max_ncorr=3,
    step_rule=MadQP.AdaptiveStep(0.99),
    regularization=MadQP.FixedRegularization(1e-8, -1e-9),
    rethrow_error=true,
)

MadQP.solve!(solver)

