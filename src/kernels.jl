function set_initial_primal_rhs!(solver::MadNLP.AbstractMadNLPSolver)
    p = solver.p
    fill!(full(p), 0.0)
    py = MadNLP.dual(p)
    b = solver.c

    py .= .- b
    return
end

function set_initial_dual_rhs!(solver::MadNLP.AbstractMadNLPSolver)
    p = solver.p
    fill!(full(p), 0.0)
    px = MadNLP.primal(p)
    c = MadNLP.primal(solver.f)

    px .= .- c
    return
end

function set_predictive_rhs!(solver::MadNLP.AbstractMadNLPSolver, kkt::MadNLP.AbstractKKTSystem)
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

function set_correction_rhs!(solver::MadNLP.AbstractMadNLPSolver, kkt::MadNLP.AbstractKKTSystem, mu::Float64, correction_lb::Vector{Float64}, correction_ub::Vector{Float64}, ind_lb, ind_ub)
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

    px .= .- f .+ zl .- zu .- solver.jacl
    py .= .- solver.c
    pzl .= (solver.xl_r .- solver.x_lr) .* solver.zl_r .+ mu .- correction_lb
    pzu .= (solver.xu_r .- solver.x_ur) .* solver.zu_r .- mu .- correction_ub
    return
end

function get_correction!(
    solver::MadNLP.AbstractMadNLPSolver,
    correction_lb,
    correction_ub,
)
    dx = MadNLP.primal(solver.d)
    dlb = MadNLP.dual_lb(solver.d)
    dub = MadNLP.dual_ub(solver.d)

    correction_lb .= solver.dx_lr .* dlb
    correction_ub .= solver.dx_ur .* dub
    return
end

# Gondzio's multi-correction scheme
function set_extra_correction!(
    solver::MadNLP.AbstractMadNLPSolver,
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

function set_aug_diagonal_reg!(kkt::MadNLP.AbstractKKTSystem{T}, solver::MadNLP.AbstractMadNLPSolver{T}) where T
    x = MadNLP.full(solver.x)
    xl = MadNLP.full(solver.xl)
    xu = MadNLP.full(solver.xu)
    zl = MadNLP.full(solver.zl)
    zu = MadNLP.full(solver.zu)

    fill!(kkt.reg, solver.del_w)
    fill!(kkt.du_diag, solver.del_c)
    kkt.l_diag .= solver.xl_r .- solver.x_lr   # (Xˡ - X)
    kkt.u_diag .= solver.x_ur .- solver.xu_r   # (X - Xᵘ)
    copyto!(kkt.l_lower, solver.zl_r)
    copyto!(kkt.u_lower, solver.zu_r)

    copyto!(kkt.pr_diag, kkt.reg)
    kkt.pr_diag[kkt.ind_lb] .-= kkt.l_lower ./ kkt.l_diag
    kkt.pr_diag[kkt.ind_ub] .-= kkt.u_lower ./ kkt.u_diag
    return
end

# Special function for ScaledSparseKKTSystem to ensure coefficients are positive
function set_aug_diagonal_reg!(kkt::MadNLP.ScaledSparseKKTSystem{T}, solver::MadNLP.AbstractMadNLPSolver{T}) where T
    x = MadNLP.full(solver.x)
    xl = MadNLP.full(solver.xl)
    xu = MadNLP.full(solver.xu)
    zl = MadNLP.full(solver.zl)
    zu = MadNLP.full(solver.zu)

    fill!(kkt.reg, solver.del_w)
    fill!(kkt.du_diag, solver.del_c)
    kkt.l_diag .= solver.x_lr .- solver.xl_r   # (X - Xˡ)
    kkt.u_diag .= solver.xu_r .- solver.x_ur   # (Xᵘ - X)
    copyto!(kkt.l_lower, solver.zl_r)
    copyto!(kkt.u_lower, solver.zu_r)

    MadNLP._set_aug_diagonal!(kkt)
    return
end

#=
    Barrier
=#

function get_complementarity_measure(solver::MadNLP.AbstractMadNLPSolver)
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

function get_affine_complementarity_measure(solver::MadNLP.AbstractMadNLPSolver, alpha_p, alpha_d)
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

function update_barrier!(rule::Mehrotra, solver, mu_affine)
    has_inequalities = (length(solver.ind_llb) + length(solver.ind_uub)) > 0
    mu_curr = get_complementarity_measure(solver)             # μ = y' s / m
    sigma = if has_inequalities
        clamp((mu_affine / mu_curr)^3, 1e-6, 10.0)
    else
        1.0
    end
    solver.mu = max(solver.opt.mu_min, sigma * mu_curr)
    return mu_curr
end

#=
    Step
=#

function get_alpha_max_primal(xl, lx, xu, ux, dxl, dxu, tau)
    alpha_xl, alpha_xu = 1.0, 1.0
    iblock_l, iblock_u = 0, 0
    @inbounds for i in eachindex(xl)
        if dxl[i] < 0 && (xl[i] + alpha_xl * dxl[i] < lx[i])
            alpha_xl =(-xl[i]+lx[i])*tau/dxl[i]
            iblock_l = i
        end
    end
    @inbounds for i in eachindex(xu)
        if dxu[i] > 0 && (xu[i] + alpha_xu * dxu[i] > ux[i])
            alpha_xu =(-xu[i]+ux[i])*tau/dxu[i]
            iblock_u = i
        end
    end
    return alpha_xl, alpha_xu, iblock_l, iblock_u
end

function get_alpha_max_dual(zl_r, zu_r, dzl, dzu, tau)
    alpha_zl, alpha_zu = 1.0, 1.0
    iblock_l, iblock_u = 0, 0
    @inbounds for i=1:length(zl_r)
        if dzl[i] < 0 && (zl_r[i] + alpha_zl * dzl[i] < 0.0)
            alpha_zl = -zl_r[i]*tau/dzl[i]
            iblock_l = i
        end
    end
    @inbounds for i=1:length(zu_r)
        if dzu[i] < 0 && (zu_r[i] + alpha_zu * dzu[i] < 0.0)
            alpha_zu = -zu_r[i]*tau/dzu[i]
            iblock_u = i
        end
    end
    return alpha_zl, alpha_zu, iblock_l, iblock_u
end

function get_fraction_to_boundary_step(solver, tau)
    alpha_xl, alpha_xu, _ = get_alpha_max_primal(
        solver.x_lr, solver.xl_r,
        solver.x_ur, solver.xu_r,
        solver.dx_lr, solver.dx_ur,
        tau,
    )
    alpha_zl, alpha_zu, _ = get_alpha_max_dual(
        solver.zl_r,
        solver.zu_r,
        MadNLP.dual_lb(solver.d),
        MadNLP.dual_ub(solver.d),
        tau,
    )
    return min(alpha_xl, alpha_xu), min(alpha_zl, alpha_zu)
end

function update_step!(rule::ConservativeStep, solver)
    alpha_p, alpha_d = get_fraction_to_boundary_step(solver, rule.tau)
    solver.alpha_p = alpha_p
    solver.alpha_d = alpha_d
    return
end

# Implement conservative rule for QP
function update_step!(rule::AdaptiveStep, solver)
    tau = max(1-solver.mu, rule.tau_min)
    alpha_p, alpha_d = get_fraction_to_boundary_step(solver, tau)
    solver.alpha_p = alpha_p
    solver.alpha_d = alpha_d
    return
end

# Implement Mehrotra's heuristic to compute the step : see Procedure GTSF (Exhibit 6.1) in
# "On The Implementation Of A Primal-Dual Interior Point Method"
function update_step!(rule::MehrotraAdaptiveStep, solver)
    gamma_a = 1.0 / (1.0 - rule.gamma_f)
    tau = 1.0

    d_zl = MadNLP.dual_lb(solver.d)
    d_zu = MadNLP.dual_ub(solver.d)

    alpha_xl, alpha_xu, i_xl, i_xu = get_alpha_max_primal(
        solver.x_lr, solver.xl_r,
        solver.x_ur, solver.xu_r,
        solver.dx_lr, solver.dx_ur, tau,
    )
    alpha_zl, alpha_zu, i_zl, i_zu = get_alpha_max_dual(
        solver.zl_r, solver.zu_r, d_zl, d_zu, tau,
    )

    max_alpha_p = min(alpha_xl, alpha_xu)
    max_alpha_d = min(alpha_zl, alpha_zu)

    mu_full = get_affine_complementarity_measure(solver, max_alpha_p, max_alpha_d)
    mu_full /= gamma_a

    alpha_p, alpha_d = 1.0, 1.0

    if max_alpha_p < 1.0
        if alpha_xl <= alpha_xu
            tmp = mu_full / (solver.zl_r[i_xl] + max_alpha_d * d_zl[i_xl])
            alpha_p = (solver.x_lr[i_xl] - solver.xl_r[i_xl] - tmp) / (-solver.dx_lr[i_xl])
        else
            tmp = mu_full / (solver.zu_r[i_xu] + max_alpha_d * d_zu[i_xu])
            alpha_p = (solver.xu_r[i_xu] - solver.x_ur[i_xu] - tmp) / (solver.dx_ur[i_xu])
        end
    end
    if max_alpha_d < 1.0
        if alpha_zl <= alpha_zu
            tmp = mu_full / (solver.x_lr[i_zl] + max_alpha_p * solver.dx_lr[i_zl] - solver.xl_r[i_zl])
            alpha_d = -(solver.zl_r[i_zl] - tmp) / d_zl[i_zl]
        else
            tmp = mu_full / (solver.xu_r[i_zu] - solver.x_ur[i_zu] - max_alpha_p * solver.dx_ur[i_zu])
            alpha_d = -(solver.zu_r[i_zu] - tmp) / d_zu[i_zu]
        end
    end

    solver.alpha_p = max(alpha_p, rule.gamma_f * max_alpha_p)
    solver.alpha_d = max(alpha_d, rule.gamma_f * max_alpha_d)
    return
end

#=
    Regularization
=#

function init_regularization!(solver::MPCSolver, ::NoRegularization)
    solver.del_w = 1.0
    solver.del_c = 0.0
    return
end

function update_regularization!(solver::MPCSolver, ::NoRegularization)
    solver.del_w = 0.0
    solver.del_c = 0.0
    return
end

function init_regularization!(solver::MPCSolver, reg::FixedRegularization)
    solver.del_w = 1.0
    solver.del_c = reg.delta_d
    return
end

function update_regularization!(solver::MPCSolver, reg::FixedRegularization)
    solver.del_w = reg.delta_p
    solver.del_c = reg.delta_d
    return
end

function init_regularization!(solver::MPCSolver, reg::AdaptiveRegularization)
    solver.del_w = 1.0
    solver.del_c = reg.delta_d
    return
end

function update_regularization!(solver::MPCSolver, reg::AdaptiveRegularization)
    reg.delta_p = max(reg.delta_p / 10.0, reg.delta_min)
    # Dual regularization is negative!
    reg.delta_d = min(reg.delta_d / 10.0, -reg.delta_min)
    solver.del_w = reg.delta_p
    solver.del_c = reg.delta_d
    return
end

#=
    Stopping criterion
=#

# Dual objective
function dual_objective(solver::MPCSolver)
    return -dot(solver.y, solver.rhs) + dot(solver.zl_r, solver.xl_r) - dot(solver.zu_r, solver.xu_r)
end

function get_optimality_gap(solver::MPCSolver, ::LinearProgram)
    primal_obj = solver.obj_val
    dual_obj = dual_objective(solver)
    return (primal_obj - dual_obj) / max(1.0, abs(dual_obj))
end

function get_optimality_gap(solver::MPCSolver, ::QuadraticProgram)
    return MadNLP.get_inf_compl(
        solver.x_lr,
        solver.xl_r,
        solver.zl_r,
        solver.xu_r,
        solver.x_ur,
        solver.zu_r,
        0.,
        1.0,
    )
end
