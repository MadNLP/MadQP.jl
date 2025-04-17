
using Test
using MadNLP
using MadQP
using MadNLPTests

function _compare_with_nlp(n, m, ind_fixed, ind_eq; max_ncorr=0, atol=1e-5)
    x0 = zeros(n)
    qp = MadNLPTests.DenseDummyQP(x0; m=m)
    nlp_solver = MadNLP.MadNLPSolver(qp; print_level=MadNLP.ERROR)
    nlp_stats = MadNLP.solve!(nlp_solver)

    qp_solver = MadNLP.MadNLPSolver(qp; print_level=MadNLP.ERROR)
    qp_stats = MadQP.solve!(qp_solver; max_ncorr=max_ncorr)

    @test qp_stats.status == MadNLP.SOLVE_SUCCEEDED
    @test qp_stats.objective ≈ nlp_stats.objective atol=atol
    @test qp_stats.solution ≈ nlp_stats.solution atol=atol
    @test qp_stats.constraints ≈ nlp_stats.constraints atol=atol
    @test qp_stats.multipliers ≈ nlp_stats.multipliers atol=atol
    return
end

@testset "Test with DenseDummyQP" begin
    @testset "Size: ($n, $m)" for (n, m) in [(10, 0), (10, 5), (50, 10)]
        _compare_with_nlp(n, m, Int[], Int[]; atol=1e-4)
    end
    @testset "Equality constraints" begin
        n, m = 20, 15
        # Default Mehrotra-predictor.
        _compare_with_nlp(n, m, Int[], Int[1, 2, 3, 8]; atol=1e-5, max_ncorr=0)
        # Gondzio's multiple correction.
        _compare_with_nlp(n, m, Int[], Int[1, 2, 3, 8]; atol=1e-5, max_ncorr=5)
    end
    @testset "Fixed variables" begin
        n, m = 20, 15
        _compare_with_nlp(n, m, Int[1, 2], Int[]; atol=1e-5)
        _compare_with_nlp(n, m, Int[1, 2], Int[1, 2, 3, 8]; atol=1e-5)
    end
end

