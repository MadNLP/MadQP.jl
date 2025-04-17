
#=
    Initialization
=#

function set_initial_primal_rhs!(solver::MadNLP.MadNLPSolver)
    px = MadNLP.primal(solver.p)
    py = MadNLP.dual(solver.p)
    c = solver.c
    f = MadNLP.full(solver.f)

    @inbounds @simd for i in eachindex(px)
        px[i] = -f[i]
    end
    @inbounds @simd for i in eachindex(py)
        py[i] = -c[i]
    end
    return
end

function init_least_square_primals!(solver::MadNLP.MadNLPSolver)
    lb, ub = solver.xl_r, solver.xu_r
    zl, zu = solver.zl_r, solver.zu_r
    xl, xu = solver.x_lr, solver.x_ur
    res = solver.jacl #

    set_initial_primal_rhs!(solver)

    # TODO: check if the linear system is correct
    solver.kkt.reg .= 1e-0
    solver.kkt.pr_diag .= 1e-0
    solver.kkt.du_diag .= -1.0e-8
    MadNLP.factorize_wrapper!(solver)
    is_solved = MadNLP.solve_refine_wrapper!(solver.d, solver, solver.p, solver._w4)
    @assert is_solved
    copyto!(MadNLP.primal(solver.x), MadNLP.primal(solver.d))
    solver.y .= MadNLP.dual(solver.d)

    zl .= solver.opt.bound_push
    zu .= solver.opt.bound_push

    delta_x = max(
        0.0,
        -1.5 * minimum(xl .- lb; init=0.0),
        -1.5 * minimum(ub .- xu; init=0.0),
    )

    delta_s = max(
        0.0,
        -1.5 * minimum(zl; init=0.0),
        -1.5 * minimum(zu; init=0.0),
    )

    xl .= xl .+ delta_x
    xu .= xu .- delta_x
    zl .+= delta_s
    zu .+= delta_s

    μ = dot(xl .- lb, zl) + dot(ub .- xu, zu)

    delta_x2 = μ / (2 * (sum(zl) + sum(zu)))
    delta_s2 = μ / (2 * (sum(xl .- lb) + sum(ub .- xu)))

    xl .+= delta_x2
    xu .-= delta_x2
    zl .+= delta_s2
    zu .+= delta_s2

    MadNLP.initialize_variables!(
        MadNLP.full(solver.x),
        MadNLP.full(solver.xl),
        MadNLP.full(solver.xu),
        solver.opt.bound_push, solver.opt.bound_fac
    )

    @assert all(solver.zl_r .> 0.0)
    @assert all(solver.zu_r .> 0.0)
    @assert all(solver.x_lr .> solver.xl_r)
    @assert all(solver.x_ur .< solver.xu_r)
    return
end

function initialize!(solver::MadNLP.MadNLPSolver{T}) where T
    opt = solver.opt
    # Initialization
    MadNLP.initialize!(
        solver.cb,
        solver.x,
        solver.xl,
        solver.xu,
        solver.y,
        solver.rhs,
        solver.ind_ineq;
        tol=opt.tol,
        bound_push=opt.bound_push,
        bound_fac=opt.bound_fac,
    )

    fill!(solver.jacl, zero(T))

    MadNLP.initialize_variables!(
        MadNLP.full(solver.x),
        MadNLP.full(solver.xl),
        MadNLP.full(solver.xu),
        opt.bound_push, opt.bound_fac
    )

    # Initializing scaling factors
    # TODO: Implement Ruiz equilibration scaling here
    if opt.nlp_scaling
        MadNLP.set_scaling!(
            solver.cb,
            solver.x,
            solver.xl,
            solver.xu,
            solver.y,
            solver.rhs,
            solver.ind_ineq,
            opt.nlp_scaling_max_gradient
        )
    end

    # Initializing KKT system
    MadNLP.initialize!(solver.kkt)

    # Initializing callbacks
    MadNLP.eval_jac_wrapper!(solver, solver.kkt, solver.x)
    MadNLP.eval_grad_f_wrapper!(solver, solver.f, solver.x)

    MadNLP.eval_cons_wrapper!(solver, solver.c, solver.x)
    solver.obj_val = MadNLP.eval_f_wrapper(solver, solver.x)
    MadNLP.eval_lag_hess_wrapper!(solver, solver.kkt, solver.x, solver.y)

    init_least_square_primals!(solver)

    theta = MadNLP.get_theta(solver.c)
    solver.theta_max = 1e4*max(1,theta)
    solver.theta_min = 1e-4*max(1,theta)
    solver.mu = opt.mu_init
    solver.tau = max(opt.tau_min,1-opt.mu_init)

    return MadNLP.REGULAR
end

#=
    Kernels
=#

function get_complementarity_measure(solver::MadNLP.MadNLPSolver)
    m1, m2 = length(solver.x_lr), length(solver.x_ur)
    if m1 + m2 == 0
        return 0.0
    end
    inf_compl = 0.0
    @inbounds @simd for i in 1:m1
        inf_compl += (solver.x_lr[i]-solver.xl_r[i])*solver.zl_r[i]
    end
    @inbounds @simd for i in 1:m2
        inf_compl += (solver.xu_r[i]-solver.x_ur[i])*solver.zu_r[i]
    end
    return inf_compl / (m1 + m2)
end

function get_affine_complementarity_measure(solver::MadNLP.MadNLPSolver, alpha_p::Float64, alpha_d::Float64)
    m1, m2 = length(solver.x_lr), length(solver.x_ur)
    if m1 + m2 == 0
        return 0.0
    end
    dzlb =  MadNLP.dual_lb(solver.d)
    dzub =  MadNLP.dual_ub(solver.d)
    @assert all(isfinite, solver.d.values)

    inf_compl = 0.0
    @inbounds @simd for i in 1:m1
        x_lb = solver.xl_r[i]
        x_ = solver.x_lr[i] + alpha_p * solver.dx_lr[i]
        z_ = solver.zl_r[i] + alpha_d * dzlb[i]
        inf_compl += (x_ - x_lb) * z_
    end
    @inbounds @simd for i in 1:m2
        x_ub = solver.xu_r[i]
        x_ = solver.x_ur[i] + alpha_p * solver.dx_ur[i]
        z_ = solver.zu_r[i] + alpha_d * dzub[i]
        inf_compl += (x_ub - x_) * z_
    end

    return inf_compl / (m1 + m2)
end

function get_alpha_max_primal(x, xl, xu, dx, tau)
    alpha_max = 1.0
    @inbounds @simd for i=1:length(x)
        dx[i]<0 && (alpha_max=min(alpha_max,(-x[i]+xl[i])*tau/dx[i]))
        dx[i]>0 && (alpha_max=min(alpha_max,(-x[i]+xu[i])*tau/dx[i]))
    end
    return alpha_max
end

function get_alpha_max_dual(zl_r, zu_r, dzl, dzu, tau)
    alpha_z = 1.0
    @inbounds @simd for i=1:length(zl_r)
        dzl[i] < 0 && (alpha_z=min(alpha_z,-zl_r[i]*tau/dzl[i]))
     end
    @inbounds @simd for i=1:length(zu_r)
        dzu[i] < 0 && (alpha_z=min(alpha_z,-zu_r[i]*tau/dzu[i]))
    end
    return alpha_z
end

function set_predictive_rhs!(solver::MadNLP.MadNLPSolver, kkt::MadNLP.AbstractKKTSystem)
    # RHS
    px = MadNLP.primal(solver.p)
    py = MadNLP.dual(solver.p)
    pzl = MadNLP.dual_lb(solver.p)
    pzu = MadNLP.dual_ub(solver.p)
    zl = MadNLP.full(solver.zl)
    zu = MadNLP.full(solver.zu)
    # Gradient
    f = MadNLP.primal(solver.f)
    # Constraint
    c = solver.c

    fill!(MadNLP.full(solver.p), 0.0)

    px  .= .-f .+ zl .- zu .- solver.jacl
    py  .= .-c
    pzl .= (solver.xl_r .- solver.x_lr) .* solver.zl_r
    pzu .= (solver.xu_r .- solver.x_ur) .* solver.zu_r
    return
end

function set_corrective_rhs!(solver::MadNLP.MadNLPSolver, kkt::MadNLP.AbstractKKTSystem, mu::Float64, correction_lb::Vector{Float64}, correction_ub::Vector{Float64}, ind_lb, ind_ub)
    px = MadNLP.primal(solver.p)
    py = MadNLP.dual(solver.p)
    pzl = MadNLP.dual_lb(solver.p)
    pzu = MadNLP.dual_ub(solver.p)

    x = MadNLP.primal(solver.x)
    f = MadNLP.primal(solver.f)
    xl = MadNLP.primal(solver.xl)
    xu = MadNLP.primal(solver.xu)
    zl = MadNLP.full(solver.zl)
    zu = MadNLP.full(solver.zu)

    px .= .-f .+ zl .- zu .- solver.jacl
    py .= .-solver.c
    pzl .= (solver.xl_r .- solver.x_lr) .* solver.zl_r .+ mu .- correction_lb
    pzu .= (solver.xu_r .- solver.x_ur) .* solver.zu_r .- mu .- correction_ub
    return
end

function get_correction!(
    solver::MadNLP.MadNLPSolver,
    correction_lb,
    correction_ub,
)
    dx = MadNLP.primal(solver.d)
    dlb = MadNLP.dual_lb(solver.d)
    dub = MadNLP.dual_ub(solver.d)

    for i in eachindex(dlb)
        correction_lb[i] = solver.dx_lr[i] * dlb[i]
    end
    for i in eachindex(dub)
        correction_ub[i] = solver.dx_ur[i] * dub[i]
    end
    return
end

# Gondzio's multi-correction scheme
function set_extra_correction!(
    solver::MadNLP.MadNLPSolver,
    correction_lb, correction_ub,
    alpha_p, alpha_d, βmin, βmax, μ,
)
    dx = MadNLP.primal(solver.d)
    dlb = MadNLP.dual_lb(solver.d)
    dub = MadNLP.dual_ub(solver.d)
    tmin, tmax = βmin * μ , βmax * μ
    # / Lower-bound
    for i in eachindex(dlb)
        x_ = solver.x_lr[i] + alpha_p * solver.dx_lr[i] - solver.xl_r[i]
        z_ = solver.zl_r[i] + alpha_d * dlb[i]
        v = x_ * z_
        correction_lb[i] -= if v < tmin
            tmin - v
        elseif v > tmax
            tmax - v
        else
            0.0
        end
    end
    # / Upper-bound
    for i in eachindex(dub)
        x_ = solver.xu_r[i] - alpha_p * solver.dx_ur[i] - solver.x_ur[i]
        z_ = solver.zu_r[i] + alpha_d * dub[i]
        v = x_ * z_
        correction_ub[i] += if v < tmin
            tmin - v
        elseif v > tmax
            tmax - v
        else
            0.0
        end
    end
    return
end

function get_fraction_to_boundary_step(solver, tau)
    alpha_p = get_alpha_max_primal(
        MadNLP.primal(solver.x),
        MadNLP.primal(solver.xl),
        MadNLP.primal(solver.xu),
        MadNLP.primal(solver.d),
        tau,
    )
    alpha_d = get_alpha_max_dual(
        solver.zl_r,
        solver.zu_r,
        MadNLP.dual_lb(solver.d),
        MadNLP.dual_ub(solver.d),
        tau,
    )
    return (alpha_p, alpha_d)
end

function set_aug_diagonal_reg!(kkt::MadNLP.AbstractKKTSystem{T}, solver::MadNLP.MadNLPSolver{T}) where T
    x = MadNLP.full(solver.x)
    xl = MadNLP.full(solver.xl)
    xu = MadNLP.full(solver.xu)
    zl = MadNLP.full(solver.zl)
    zu = MadNLP.full(solver.zu)

    # TODO: implement primal-dual regularization here
    fill!(kkt.reg, 0.0)
    fill!(kkt.du_diag, 0)
    kkt.l_diag .= solver.xl_r .- solver.x_lr   # (Xˡ - X)
    kkt.u_diag .= solver.x_ur .- solver.xu_r   # (X - Xᵘ)
    copyto!(kkt.l_lower, solver.zl_r)
    copyto!(kkt.u_lower, solver.zu_r)

    copyto!(kkt.pr_diag, kkt.reg)
    kkt.pr_diag[kkt.ind_lb] .-= kkt.l_lower ./ kkt.l_diag
    kkt.pr_diag[kkt.ind_ub] .-= kkt.u_lower ./ kkt.u_diag
    return
end

#=
    Algorithm
=#

function affine_step!(solver::MadNLP.AbstractMadNLPSolver)
    set_predictive_rhs!(solver, solver.kkt)
    is_solved = MadNLP.solve_refine_wrapper!(solver.d, solver, solver.p, solver._w4)
    return get_fraction_to_boundary_step(solver, 1.0)
end

function mehrotra_correction_step!(solver, correction_lb, correction_ub, ind_cons)
    set_corrective_rhs!(solver, solver.kkt, solver.mu, correction_lb, correction_ub, ind_cons.ind_lb, ind_cons.ind_ub)
    is_solved = MadNLP.solve_refine_wrapper!(solver.d, solver, solver.p, solver._w4)
    solver.tau = max(1-solver.mu, solver.opt.tau_min)
    return get_fraction_to_boundary_step(solver, solver.tau)
end

function gondzio_correction_step!(solver, correction_lb, correction_ub, ind_cons, mu_curr, max_ncorr)
    δ = 0.1
    γ = 0.1
    βmin = 0.1
    βmax = 10.0
    # Load buffer for descent direction.
    Δp = solver._w3.values

    # TODO: this may be redundant with (alpha_p, alpha_d) computed in Mehrotra correction step.
    alpha_p, alpha_d = get_fraction_to_boundary_step(solver, solver.tau)

    for ncorr in 1:max_ncorr
        # Enlarge step sizes in primal and dual spaces.
        tilde_alpha_p = min(alpha_p + δ, 1.0)
        tilde_alpha_d = min(alpha_d + δ, 1.0)
        # Apply Mehrotra's heuristic for centering parameter mu.
        ga = get_affine_complementarity_measure(solver, tilde_alpha_p, tilde_alpha_d)
        g = mu_curr
        mu = (ga / g)^2 * ga                    # Eq. (12)
        # Add additional correction.
        set_extra_correction!(
            solver, correction_lb, correction_ub,
            tilde_alpha_p, tilde_alpha_d, βmin, βmax, mu,
        )
        # Update RHS.
        set_corrective_rhs!(
            solver,
            solver.kkt,
            mu,
            correction_lb,
            correction_ub,
            ind_cons.ind_lb,
            ind_cons.ind_ub,
        )
        # Solve KKT linear system.
        copyto!(Δp, solver.d.values)
        MadNLP.solve_refine_wrapper!(solver.d, solver, solver.p, solver._w4)
        hat_alpha_p, hat_alpha_d = get_fraction_to_boundary_step(solver, solver.tau)

        # Stop extra correction if the stepsize does not increase sufficiently
        if (hat_alpha_p < 1.005 * alpha_p) || (hat_alpha_d < 1.005 * alpha_d)
            copyto!(solver.d.values, Δp)
            break
        else
            alpha_p = hat_alpha_p
            alpha_d = hat_alpha_d
        end
    end

    return alpha_p, alpha_d
end

# Predictor-corrector method
function mpc!(solver::MadNLP.AbstractMadNLPSolver; max_ncorr=0)
    ind_cons = MadNLP.get_index_constraints(
        solver.nlp;
        fixed_variable_treatment=solver.opt.fixed_variable_treatment,
        equality_treatment=solver.opt.equality_treatment,
    )
    nlb, nub = length(ind_cons.ind_lb), length(ind_cons.ind_ub)
    has_inequalities = (nlb + nub > 0)

    # Buffers for multiple correction scheme.
    correction_lb = zeros(nlb)
    correction_ub = zeros(nub)

    while true
        # A' y
        MadNLP.jtprod!(solver.jacl, solver.kkt, solver.y)

        #####
        # Update info
        #####
        sd = MadNLP.get_sd(solver.y,solver.zl_r,solver.zu_r,solver.opt.s_max)
        sc = MadNLP.get_sc(solver.zl_r,solver.zu_r,solver.opt.s_max)
        solver.inf_pr = MadNLP.get_inf_pr(solver.c)
        solver.inf_du = MadNLP.get_inf_du(
            MadNLP.full(solver.f),
            MadNLP.full(solver.zl),
            MadNLP.full(solver.zu),
            solver.jacl,
            1.0,
        )
        solver.inf_compl = MadNLP.get_inf_compl(solver.x_lr,solver.xl_r,solver.zl_r,solver.xu_r,solver.x_ur,solver.zu_r,0.,sc)
        MadNLP.print_iter(solver)

        #####
        # Termination criteria
        #####
        if max(solver.inf_pr,solver.inf_du,solver.inf_compl) <= solver.opt.tol
            return MadNLP.SOLVE_SUCCEEDED
        elseif solver.cnt.k >= solver.opt.max_iter
            return MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
        end

        #####
        # Factorize
        #####
        set_aug_diagonal_reg!(solver.kkt, solver)
        # MadNLP.set_aug_diagonal!(solver.kkt, solver)
        MadNLP.factorize_wrapper!(solver)

        #####
        # Prediction step
        #####
        alpha_p, alpha_d = affine_step!(solver)
        alpha_aff = min(alpha_p, alpha_d)
        mu_affine = get_affine_complementarity_measure(solver, alpha_aff, alpha_aff)
        get_correction!(solver, correction_lb, correction_ub)

        #####
        # Update barrier
        #####
        mu_curr = get_complementarity_measure(solver)             # μ = y' s / m
        sigma = if has_inequalities
            clamp((mu_affine / mu_curr)^3, 1e-6, 10.0)
        else
            1.0
        end
        solver.mu = max(solver.opt.mu_min, sigma * mu_curr)

        #####
        # Mehrotra's Correction step
        #####
        alpha_p, alpha_d = mehrotra_correction_step!(
            solver,
            correction_lb,
            correction_ub,
            ind_cons,
        )

        #####
        # Gondzio's additional correction
        #####
        if max_ncorr > 0
            alpha_p, alpha_d = gondzio_correction_step!(
                solver,
                correction_lb,
                correction_ub,
                ind_cons,
                mu_curr,
                max_ncorr,
            )
        end

        #####
        # Next trial point
        #####
        solver.alpha = min(alpha_p, alpha_d)
        solver.alpha_z = solver.alpha
        # Update primal-dual solution
        axpy!(solver.alpha, MadNLP.primal(solver.d), MadNLP.primal(solver.x))
        axpy!(solver.alpha, MadNLP.dual(solver.d), solver.y)

        solver.zl_r .+= solver.alpha_z .* MadNLP.dual_lb(solver.d)
        solver.zu_r .+= solver.alpha_z .* MadNLP.dual_ub(solver.d)

        solver.obj_val = MadNLP.eval_f_wrapper(solver, solver.x)
        MadNLP.eval_cons_wrapper!(solver, solver.c, solver.x)
        MadNLP.eval_grad_f_wrapper!(solver, solver.f, solver.x)

        adjusted = MadNLP.adjust_boundary!(solver.x_lr,solver.xl_r,solver.x_ur,solver.xu_r,solver.mu)

        solver.cnt.k+=1
        solver.cnt.l=1
    end
end

function solve!(
    solver::MadNLP.MadNLPSolver;
    max_ncorr=0,
    kwargs...
)
    stats = MadNLP.MadNLPExecutionStats(solver)
    nlp = solver.nlp
    solver.cnt.start_time = time()

    if !isempty(kwargs)
        MadNLP.set_options!(solver.opt, kwargs)
    end

    try
        MadNLP.@notice(solver.logger,"This is MadLP, running with $(MadNLP.introduce(solver.kkt.linear_solver))\n")
        MadNLP.print_init(solver)
        initialize!(solver)
        solver.status = mpc!(solver; max_ncorr=max_ncorr)
    catch e
        if e isa MadNLP.InvalidNumberException
            if e.callback == :obj
                solver.status=MadNLP.INVALID_NUMBER_OBJECTIVE
            elseif e.callback == :grad
                solver.status=MadNLP.INVALID_NUMBER_GRADIENT
            elseif e.callback == :cons
                solver.status=MadNLP.INVALID_NUMBER_CONSTRAINTS
            elseif e.callback == :jac
                solver.status=MadNLP.INVALID_NUMBER_JACOBIAN
            elseif e.callback == :hess
                solver.status=MadNLP.INVALID_NUMBER_HESSIAN_LAGRANGIAN
            else
                solver.status=MadNLP.INVALID_NUMBER_DETECTED
            end
        elseif e isa MadNLP.NotEnoughDegreesOfFreedomException
            solver.status=MadNLP.NOT_ENOUGH_DEGREES_OF_FREEDOM
        elseif e isa MadNLP.LinearSolverException
            solver.status=MadNLP.ERROR_IN_STEP_COMPUTATION;
            solver.opt.rethrow_error && rethrow(e)
        elseif e isa InterruptException
            solver.status=MadNLP.USER_REQUESTED_STOP
            solver.opt.rethrow_error && rethrow(e)
        else
            solver.status=MadNLP.INTERNAL_ERROR
            solver.opt.rethrow_error && rethrow(e)
        end
    finally
        solver.cnt.total_time = time() - solver.cnt.start_time
        if !(solver.status < MadNLP.SOLVE_SUCCEEDED)
            MadNLP.print_summary(solver)
        end
        MadNLP.@notice(solver.logger,"EXIT: $(MadNLP.get_status_output(solver.status, solver.opt))")
        solver.opt.disable_garbage_collector &&
            (GC.enable(true); MadNLP.@warn(solver.logger,"Julia garbage collector is turned back on"))
        finalize(solver.logger)

        MadNLP.update!(stats,solver)
    end

    return stats
end

