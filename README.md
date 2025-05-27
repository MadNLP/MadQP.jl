# MadIPM.jl

MadIPM.jl is an extension of MadNLP.jl for linear and quadratic programming.

It implements the Mehrotra predictor-corrector method, leading
to faster convergence than the default filter line-search algorithm
used in MadNLP.

MadIPM.jl is built on top of MadNLP.jl, and makes an extensive
use of the `AbstractKKTSystem` abstraction for modularity.
