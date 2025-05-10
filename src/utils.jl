
abstract type AbstractConicProblem end

struct LinearProgram <: AbstractConicProblem end
struct QuadraticProgram <: AbstractConicProblem end

#=
    Barrier update
=#

abstract type AbstractBarrierUpdate end
struct Mehrotra <: AbstractBarrierUpdate end

#=
    Step rule for next iterate
=#

abstract type AbstractStepRule end

@kwdef struct ConservativeStep{T} <: AbstractStepRule
    tau::T = T(0.995)
end

@kwdef struct AdaptiveStep{T} <: AbstractStepRule
    tau_min::T = T(0.99)
end

@kwdef struct MehrotraAdaptiveStep{T} <: AbstractStepRule
    gamma_f::T = T(0.99)
end

#=
    Primal-dual regularization for KKT system
=#

abstract type AbstractRegularization end

struct NoRegularization <: AbstractRegularization end

mutable struct FixedRegularization{T} <: AbstractRegularization
    delta_p::T
    delta_d::T
end

mutable struct AdaptiveRegularization{T} <: AbstractRegularization
    delta_p::T
    delta_d::T
    delta_min::T
end

#=
    Utils for linear solvers
=#

function is_factorized(::MadNLP.AbstractLinearSolver)
    return true # assume the system is factorized by default
end
function is_factorized(lin_solver::MadNLP.LDLSolver)
    return LDLFactorizations.factorized(lin_solver.inner)
end
function is_factorized(lin_solver::MadNLP.CHOLMODSolver)
    return issuccess(lin_solver.inner)
end


#=
    Options
=#

@kwdef mutable struct IPMOptions <: MadNLP.AbstractOptions
    tol::Float64
    kkt_system::Type
    linear_solver::Type
    iterator::Type = MadNLP.RichardsonIterator
    # Output options
    output_file::String = ""
    print_level::MadNLP.LogLevels = MadNLP.INFO
    file_print_level::MadNLP.LogLevels = MadNLP.INFO
    rethrow_error::Bool = false
    # Termination options
    max_iter::Int = 3000
    kappa_d::Float64 = 1e-5
    fixed_variable_treatment::Type = kkt_system <: MadNLP.SparseCondensedKKTSystem ? MadNLP.RelaxBound : MadNLP.MakeParameter
    equality_treatment::Type = kkt_system <: MadNLP.SparseCondensedKKTSystem ? MadNLP.RelaxEquality : MadNLP.EnforceEquality
    # initialization options
    scaling::Bool = true
    bound_push::Float64 = 1e-2
    bound_fac::Float64 = 1e-2
    bound_relax_factor::Float64 = 1e-8
    # Regularization
    regularization::AbstractRegularization = FixedRegularization(1e-8, 0.0)
    # Step
    step_rule::AbstractStepRule = AdaptiveStep(0.99)
    # Barrier
    barrier_update::AbstractBarrierUpdate = Mehrotra()
    max_ncorr::Int = 0
    s_max::Float64 = 100.0
    mu_init::Float64 = 1e-1
    mu_min::Float64 = 1e-11
    mu_superlinear_decrease_power::Float64 = 1.5
    tau_min::Float64 = 0.99
end

# smart option presets
function IPMOptions(
    nlp::NLPModels.AbstractNLPModel{T};
    kkt_system =  MadNLP.SparseKKTSystem,
    linear_solver =  MadNLP.default_sparse_solver(nlp),
    tol = 1e-8,
) where T
    return IPMOptions(
        tol = tol,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
    )
end

function load_options(nlp; options...)
    primary_opt, options = MadNLP._get_primary_options(options)

    # Initiate interior-point options
    opt_ipm = IPMOptions(nlp; primary_opt...)
    linear_solver_options = MadNLP.set_options!(opt_ipm, options)
    # Initiate linear-solver options
    opt_linear_solver = MadNLP.default_options(opt_ipm.linear_solver)
    iterator_options = MadNLP.set_options!(opt_linear_solver, linear_solver_options)
    # Initiate iterator options
    opt_iterator = MadNLP.default_options(opt_ipm.iterator, opt_ipm.tol)
    remaining_options = MadNLP.set_options!(opt_iterator, iterator_options)

    # Initiate logger
    logger = MadNLP.MadNLPLogger(
        print_level=opt_ipm.print_level,
        file_print_level=opt_ipm.file_print_level,
        file = opt_ipm.output_file == "" ? nothing : open(opt_ipm.output_file,"w+"),
    )
    MadNLP.@trace(logger,"Logger is initialized.")

    # Print remaning options (unsupported)
    if !isempty(remaining_options)
        MadNLP.print_ignored_options(logger, remaining_options)
    end
    return (
        interior_point=opt_ipm,
        linear_solver=opt_linear_solver,
        iterative_refinement=opt_iterator,
        logger=logger,
    )
end

function ruiz_scaling!(cb::MadNLP.SparseCallback, x, xl, xu, y0, rhs, ind_ineq, nlp_scaling_max_gradient)
    # Model and buffers
    nlp = cb.nlp
    con_buffer = cb.con_buffer
    jac_buffer = cb.jac_buffer

    # Check with François if it is not already initialized
    x0 = MadNLP.variable(x)
    NLPModels.jac_coord!(nlp, x0, jac_buffer)

    # Compute Ruiz scaling
    Dr, Dc = HSL.mc77(cb.ncon, cb.nvar, cb.jac_I, cb.jac_J, jac_buffer, 0; symmetric=false)

    # Equilibrate the Jacobian -- J_scaled = inv(Dr) * J * inv(Dc)
    @inbounds for k in eachindex(jac_buffer)
        i = cb.jac_I[k]
        j = cb.jac_J[k]
        jac_buffer[k] = jac_buffer[k] / ( Dr[i] * Dc[j] )
    end

    # Store Dr and Dc in the solver
    nlp.row_scaling = Dr
    nlp.col_scaling = Dc

    # Apply scaling to constraints and multipliers
    NLPModels.cons!(nlp, x0, cb.con_buffer)
    cb.con_buffer .*= Dr
    rhs .= cb.con_buffer
    y0 ./= Dr

    # Scale slacks and their bounds if present
    if !isempty(ind_ineq)
        s = slack(x)
        sl = slack(xl)
        su = slack(xu)
        sc = Dr[ind_ineq]
        s .*= sc
        sl .*= sc
        su .*= sc
    end

    return
end
