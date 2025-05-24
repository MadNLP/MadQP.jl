using CUDA
using CUDA.CUSPARSE
using MadNLPGPU
using KernelAbstractions
using SparseArrays
using LinearAlgebra
import LinearAlgebra: BlasFloat

@kernel function _transfer_to_map!(dest, to_map, src)
    k = @index(Global, Linear)
    @inbounds begin
        # TODO: do we need Atomix?
        dest[to_map[k]] += src[k]
    end
end

function MadNLP.transfer!(
    dest::CUSPARSE.CuSparseMatrixCSC{Tv},
    src::MadNLP.SparseMatrixCOO{Tv},
    map::CuVector{Int},
) where {Tv}
    fill!(nonzeros(dest), zero(Tv))
    if length(map) > 0
        _transfer_to_map!(CUDABackend())(nonzeros(dest), map, src.V; ndrange=length(map))
        KernelAbstractions.synchronize(CUDABackend())
    end
    return
end

function MadNLP.compress_hessian!(
    kkt::MadNLP.SparseKKTSystem{T,VT,MT},
) where {T,VT,MT<:CUSPARSE.CuSparseMatrixCSC{T,Int32}}
    MadNLP.transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)
end

mutable struct MadQPOperator{T,M,M2} <: AbstractMatrix{T}
    type::Type{T}
    m::Int
    n::Int
    A::M
    mat::M2
    transa::Char
    descA::CUSPARSE.CuSparseMatrixDescriptor
    buffer::CuVector{UInt8}
end

Base.eltype(A::MadQPOperator{T}) where T = T
Base.size(A::MadQPOperator) = (A.m, A.n)
SparseArrays.nnz(A::MadQPOperator) = nnz(A.A)

for (SparseMatrixType, BlasType) in ((:(CuSparseMatrixCSR{T}), :BlasFloat),
                                     (:(CuSparseMatrixCSC{T}), :BlasFloat),
                                     (:(CuSparseMatrixCOO{T}), :BlasFloat))
    @eval begin
        function MadQPOperator(A::$SparseMatrixType; transa::Char='N', symmetric::Bool=false) where T <: $BlasType
            m, n = size(A)
            alpha = Ref{T}(one(T))
            beta = Ref{T}(zero(T))
            bool = symmetric && (nnz(A) > 0)
            mat = bool ? tril(A, -1) + A' : A
            descA = CUSPARSE.CuSparseMatrixDescriptor(mat, 'O')
            descX = CUSPARSE.CuDenseVectorDescriptor(T, n)
            descY = CUSPARSE.CuDenseVectorDescriptor(T, m)
            algo = CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT
            buffer_size = Ref{Csize_t}()
            CUSPARSE.cusparseSpMV_bufferSize(CUSPARSE.handle(), transa, alpha, descA, descX, beta, descY, T, algo, buffer_size)
            buffer = CuVector{UInt8}(undef, buffer_size[])
            if CUSPARSE.version() ≥ v"12.3"
                CUSPARSE.cusparseSpMV_preprocess(CUSPARSE.handle(), transa, alpha, descA, descX, beta, descY, T, algo, buffer)
            end
            M = typeof(A)
            M2 = typeof(mat)
            return MadQPOperator{T,M,M2}(T, m, n, A, mat, transa, descA, buffer)
        end
    end
end

function LinearAlgebra.mul!(y::CuVector{T}, A::MadQPOperator{T}, x::CuVector{T}) where T <: BlasFloat
    (length(y) != A.m) && throw(DimensionMismatch("length(y) != A.m"))
    (length(x) != A.n) && throw(DimensionMismatch("length(x) != A.n"))
    descY = CUSPARSE.CuDenseVectorDescriptor(y)
    descX = CUSPARSE.CuDenseVectorDescriptor(x)
    algo = CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT
    alpha = Ref{T}(one(T))
    beta = Ref{T}(zero(T))
    CUSPARSE.cusparseSpMV(CUSPARSE.handle(), A.transa, alpha, A.descA, descX, beta, descY, T, algo, A.buffer)
end

# Should be backported in MadNLPGPU.jl in the future
@kernel function _scale_augmented_system_coo_kernel!(dest_V, @Const(src_I), @Const(src_J), @Const(src_V), @Const(scaling), @Const(n), @Const(m))
    k = @index(Global, Linear)
    i = src_I[k]
    j = src_J[k]

    # Primal regularization pr_diag
    if k <= n
        dest_V[k] = src_V[k]
    # Hessian block
    elseif i <= n && j <= n
        dest_V[k] = src_V[k] * scaling[i] * scaling[j]
    # Jacobian block
    elseif n + 1 <= i <= n + m && j <= n
        dest_V[k] = src_V[k] * scaling[j]
    # Dual regularization du_diag
    elseif (n + 1 <= i <= n + m) && (n + 1 <= j <= n + m)
        dest_V[k] = src_V[k]
    end
    nothing
end

function MadNLP._build_scale_augmented_system_coo!(dest, src, scaling::CuArray, n, m)
    backend = CUDABackend()
    kernel! = _scale_augmented_system_coo_kernel!(backend)
    N = nnz(src)
    kernel!(dest.V, src.I, src.J, src.V, scaling, n, m; ndrange = N)
end
