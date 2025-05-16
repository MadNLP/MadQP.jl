
using QuadraticModels
using NLPModels
using QPSReader

function NLPModels.cons!(qp::QuadraticModel{T}, x::AbstractVector{T}, c::AbstractVector{T}) where T
    return NLPModels.cons_lin!(qp, x, c)
end

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

