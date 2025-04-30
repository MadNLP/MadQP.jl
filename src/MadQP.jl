module MadQP

using Printf
using LinearAlgebra
import MadNLP
import MadNLP: full
import NLPModels

include("utils.jl")
include("structure.jl")
include("kernels.jl")
include("solver.jl")

end # module MadQP
