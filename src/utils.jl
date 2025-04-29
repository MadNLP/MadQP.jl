
abstract type AbstractConicProblem end

struct LinearProgram <: AbstractConicProblem end
struct QuadraticProgram <: AbstractConicProblem end

abstract type AbstractBarrierUpdate end
struct Mehrotra <: AbstractBarrierUpdate end

abstract type AbstractStepRule end
struct PrimalDualStep <: AbstractStepRule end
struct ConservativeStep <: AbstractStepRule end


@kwdef mutable struct IPMOptions <: MadNLP.AbstractOptions
    tol::Float64
    kkt_system::Type
    linear_solver::Type
    iterator::Type = MadNLP.RichardsonIterator
    # Output options
    output_file::String = ""
    print_level::MadNLP.LogLevels = MadNLP.INFO
    file_print_level::MadNLP.LogLevels = MadNLP.INFO
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
    # Step
    step_rule::AbstractStepRule = PrimalDualStep()
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

