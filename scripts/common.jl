using QuadraticModels
using QPSReader
using GZip
using CodecBzip2
using HSL
using NLPModels
using SparseArrays

import QuadraticModels: SparseMatrixCOO

"""
    import_mps(filename::String)

Import instance from the file whose path is specified in `filename`.

The function parses the file's extension to adapt the import. If the extension
is `.mps`, `.sif` or `.SIF`, it directly reads the file. If the extension
is `.gz` or `.bz2`, it decompresses the file using gzip or bzip2, respectively.

"""
function import_mps(filename)
    ext = match(r"(.*)\.(.*)", filename).captures[2]
    data = if ext ∈ ("mps", "sif", "SIF")
        readqps(filename)
    elseif ext == "gz"
        GZip.open(filename, "r") do gz
            readqps(gz)
        end
    elseif ext == "bz2"
        open(filename, "r") do io
            stream = Bzip2DecompressorStream(io)
            readqps(stream)
        end
    end
    return data
end

function _scale_coo!(A, Dr, Dc)
    k = 1
    for (i, j) in zip(A.rows, A.cols)
        A.vals[k] = A.vals[k] / (Dr[i] * Dc[j])
        k += 1
    end
end

"""
    scale_qp(qp::QuadraticModel)

Scale QP using Ruiz' equilibration method.

The function scales the Jacobian ``A`` as ``As = Dr * A * Dc``, with ``As``
a matrix whose rows and columns have an infinite norm close to 1.

The scaling is computed using `HSL.mc77`, implementing the Ruiz equilibration method.

"""
function scale_qp(qp::QuadraticModel)
    A = qp.data.A
    m, n = size(A)

    if !LIBHSL_isfunctional()
        return qp
    end

    A_csc = sparse(A.rows, A.cols, A.vals, m, n)
    Dr, Dc = HSL.mc77(A_csc, 0)

    Hs = copy(qp.data.H)
    As = copy(qp.data.A)
    _scale_coo!(Hs, Dc, Dc)
    _scale_coo!(As, Dr, Dc)

    data = QuadraticModels.QPData(
        qp.data.c0,
        qp.data.c ./ Dc,
        qp.data.v,
        Hs,
        As,
    )

    return QuadraticModel(
        NLPModelMeta(
            qp.meta.nvar;
            ncon=qp.meta.ncon,
            lvar=qp.meta.lvar .* Dc,
            uvar=qp.meta.uvar .* Dc,
            lcon=qp.meta.lcon ./ Dr,
            ucon=qp.meta.ucon ./ Dr,
            x0=qp.meta.x0 .* Dc,
            y0=qp.meta.y0 ./ Dr,
            nnzj=qp.meta.nnzj,
            lin_nnzj=qp.meta.nnzj,
            lin=qp.meta.lin,
            nnzh=qp.meta.nnzh,
            minimize=qp.meta.minimize,
        ),
        Counters(),
        data,
    )
end

"""
    presolved_qp, flag = presolve_qp(qp::QuadraticModel)

Run basic presolve routines implemented in `QuadraticModels.presolve` and return
a new QuadraticModel if flag is `true`.
If `flag` is `false`, the initial `qp` is returned.
"""
function presolve_qp(qp::QuadraticModel)
    # Use routine implemented in QuadraticModels
    res = presolve(qp)

    qp_presolved = res.solver_specific[:presolvedQM]
    if qp_presolved != nothing
        new_qp = QuadraticModel(
            qp_presolved.meta,
            qp_presolved.counters,
            qp_presolved.data,
        )
        resize!(new_qp.data.v, NLPModels.get_nvar(new_qp))
        return new_qp, true
    else
        # unbounded, infeasible or nvarps == 0
        return qp, false
    end
end

"""
    standard_form_qp(qp::QuadraticModel)

Reformulate a QP into a (partially) standard form.

The function takes as input a generic QP of the form:
```
min_{x}  c'x
  s.t.   xl <= x <= xu
         bl <= Ax <= bu

```
and reformulates it by:
- introducing slack variables `s` such that `s = Ax` for inequality constraints,
- rewriting upper bounds (on `x` and `s`) as equality constraints using nonnegative slack variables `w = (wx, ws)`.

The resulting problem is equivalent to:
```
min_{x,s,w}  c'x
  s.t.       s = Ax
             x + wx = xu  (for upper-bounded x)
             s + ws = bu  (for upper-bounded Ax)
             xl <= x
             bl <= s
             0 <= w
```
Equality constraints are preserved as-is.
"""
function standard_form_qp(qp::QuadraticModel)
    n = NLPModels.get_nvar(qp)
    m = NLPModels.get_ncon(qp)

    lvar, uvar = NLPModels.get_lvar(qp), NLPModels.get_uvar(qp)
    lcon, ucon = NLPModels.get_lcon(qp), NLPModels.get_ucon(qp)

    # Count ineq. constraints
    ind_ineq = Int[]
    for i in 1:m
        (lcon[i] < ucon[i]) && push!(ind_ineq, i)
    end
    ns = length(ind_ineq)

    # Count upper-bounds in x
    ind_rng = Int[]
    ind_only_ub = Int[]
    ind_fixed = Int[]
    xu = Float64[]
    for i in 1:n
        if (lvar[i] == uvar[i])
            # Fixed variable
            push!(ind_fixed, i)
        elseif (-Inf < lvar[i] < uvar[i] < Inf)
            # Range bounds
            push!(ind_rng, i)
            push!(xu, uvar[i])
        elseif (uvar[i] < Inf)
            # Only upper bounds
            push!(ind_only_ub, i)
        end
    end

    # Count upper-bounds in future slack s
    for (k, i) in enumerate(ind_ineq)
        if (-Inf < lcon[i] < ucon[i] < Inf)
            # Range bounds
            push!(ind_rng, k + n)
            push!(xu, ucon[i])
        elseif (ucon[i] < Inf)
            # Only upper bounds
            push!(ind_only_ub, k + n)
        end
    end

    # Build problem
    nw = length(ind_rng)
    nvar = n + ns + nw
    ncon = m + nw

    # Build A and H
    Hs = SparseMatrixCOO(nvar, nvar, qp.data.H.rows, qp.data.H.cols, qp.data.H.vals)

    Ai, Aj, Ax = findnz(qp.data.A)
    Bi, Bj, Bx = similar(Ai, ns+2*nw), similar(Aj, ns+2nw), similar(Ax, ns+2*nw)

    cnt = 1
    # Slack contribution Ax - s = 0
    for (k, i) in enumerate(ind_ineq)
        Bi[cnt] = i
        Bj[cnt] = n + k
        Bx[cnt] = -1.0
        cnt += 1
    end
    # Range reformulation x + w = xu
    for (k, i) in enumerate(ind_rng)
        Bi[cnt] = m + k
        Bj[cnt] = i
        Bx[cnt] = 1.0
        cnt += 1
        Bi[cnt] = m + k
        Bj[cnt] = k + n + ns
        Bx[cnt] = 1.0
        cnt += 1
    end

    As = SparseMatrixCOO(ncon, nvar, [Ai; Bi], [Aj; Bj], [Ax; Bx])

    # Build constraints' lower and upper bounds
    lcon_ = zeros(ncon)
    ucon_ = zeros(ncon)

    for i in 1:m
        if lcon[i] < ucon[i]
            # inequality constraints
            lcon_[i] = 0.0
            ucon_[i] = 0.0
        else
            # equality constraints
            lcon_[i] = lcon[i]
            ucon_[i] = ucon[i]
        end
    end
    for (k, i) in enumerate(ind_rng)
        lcon_[m + k] = xu[k]
        ucon_[m + k] = xu[k]
    end

    lvar_ = [lvar; lcon[ind_ineq]; zeros(nw)]
    uvar_ = [uvar; ucon[ind_ineq]; fill(Inf, nw)]
    # The upper bounds in range constraints have been moved in a separate constraint
    uvar_[ind_rng] .= Inf
    # Keep fixed variables in the formulation
    uvar_[ind_fixed] .= uvar[ind_fixed]

    data = QuadraticModels.QPData(
        qp.data.c0,
        [qp.data.c; zeros(ns + nw)],
        [qp.data.v; zeros(ns + nw)],
        Hs,
        As,
    )

    return QuadraticModel(
        NLPModelMeta(
            nvar;
            ncon=ncon,
            lvar=lvar_,
            uvar=uvar_,
            lcon=lcon_,
            ucon=ucon_,
            x0=[qp.meta.x0; zeros(ns + nw)],
            y0=[qp.meta.y0; zeros(nw)],
            nnzj=qp.meta.nnzj + ns + 2*nw,
            lin_nnzj=qp.meta.nnzj + ns + 2*nw,
            lin=[qp.meta.lin; (m+1:m+nw)],
            nnzh=qp.meta.nnzh,
            minimize=qp.meta.minimize,
        ),
        Counters(),
        data,
    )
end
