using QuadraticModels
using QPSReader
using GZip
using CodecBzip2
using HSL

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
    scale_qp

Scale QP using Ruiz' equilibration method.

"""
function scale_qp(qp)
    A = qp.data.A
    m, n = size(A)

    if LIBHSL_isfunctional()
        A_csc = sparse(A.rows, A.cols, A.vals, m, n)
        Dr, Dc = HSL.mc77(A_csc, 0)

        Hs = copy(qp.data.H)
        As = copy(qp.data.A)
        _scale_coo!(Hs, Dr, Dc)
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
        ), Dr, Dc
    else
        @info("The official version of HSL_jll.jl is not installed.")
        Dr = ones(m)
        Dc = ones(n)
        return qp, Dr, Dc
    end
end

"""
    presolve_qp

Run basic presolve routines implemented in QuadraticModels and return
a new QuadraticModel.

"""
function presolve_qp(qp)
    # Use routine implemented in QuadraticModels
    res = presolve(qp)

    qp_presolved = res.solver_specific[:presolvedQM]

    new_qp = QuadraticModel(
        qp_presolved.meta,
        qp_presolved.counters,
        qp_presolved.data,
    )

    return new_qp
end
