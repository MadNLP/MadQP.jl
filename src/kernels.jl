

function set_initial_primal_rhs!(solver::MadNLP.AbstractMadNLPSolver)
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

    px .= .-f .+ zl .- zu .- solver.jacl
    py .= .-solver.c
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

    # TODO: implement primal-dual regularization here
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

function update_step!(rule::PrimalDualStep, solver, alpha_p, alpha_d)
    solver.alpha_p = alpha_p
    solver.alpha_d = alpha_d
    return
end

# Implement conservative rule for QP
function update_step!(rule::ConservativeStep, solver, alpha_p, alpha_d)
    solver.alpha_p = min(alpha_p, alpha_d)
    solver.alpha_d = solver.alpha_p
    return
end

