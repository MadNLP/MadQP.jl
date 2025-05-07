
using Test
using MadNLP
using MadQP
using MadNLPTests

function _compare_with_nlp(n, m, ind_fixed, ind_eq; max_ncorr=0, atol=1e-5)
    x0 = zeros(n)
    qp = MadNLPTests.DenseDummyQP(x0; m=m)
    nlp_solver = MadNLP.MadNLPSolver(qp; print_level=MadNLP.ERROR)
    nlp_stats = MadNLP.solve!(nlp_solver)

    qp_solver = MadQP.MPCSolver(qp; print_level=MadNLP.ERROR, max_ncorr=max_ncorr)
    qp_stats = MadQP.solve!(qp_solver)

    @test qp_stats.status == MadNLP.SOLVE_SUCCEEDED
    @test qp_stats.objective ≈ nlp_stats.objective atol=atol
    @test qp_stats.solution ≈ nlp_stats.solution atol=atol
    @test qp_stats.constraints ≈ nlp_stats.constraints atol=atol
    @test qp_stats.multipliers ≈ nlp_stats.multipliers atol=atol
    return
end

@testset "Test with DenseDummyQP" begin
    # Test results match with MadNLP
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

    # Test inner working in MadQP
    n, m = 10, 5
    x0 = zeros(n)
    qp = MadNLPTests.DenseDummyQP(x0; m=m)

    @testset "Step rule $rule" for rule in [
        MadQP.AdaptiveStep(0.99),
        MadQP.ConservativeStep(0.99),
        MadQP.MehrotraAdaptiveStep(0.99),
    ]
        qp_solver = MadQP.MPCSolver(
            qp;
            print_level=MadNLP.ERROR,
            step_rule=rule,
        )
        qp_stats = MadQP.solve!(qp_solver)
        @test qp_stats.status == MadNLP.SOLVE_SUCCEEDED
    end

    # Compute reference solution
    qp_solver = MadQP.MPCSolver(
        qp;
        print_level=MadNLP.ERROR,
        regularization=MadQP.NoRegularization(),
    )
    sol_ref = MadQP.solve!(qp_solver)

    @testset "K2.5 KKT linear system" begin
        qp_k25 = MadQP.MPCSolver(
            qp;
            print_level=MadNLP.ERROR,
            kkt_system=MadNLP.ScaledSparseKKTSystem,
        )
        sol_k25 = MadQP.solve!(qp_k25)
        @test sol_k25.status == MadNLP.SOLVE_SUCCEEDED
        @test sol_k25.iter ≈ sol_ref.iter atol=1e-6
        @test sol_k25.objective ≈ sol_ref.objective atol=1e-6
        @test sol_k25.solution ≈ sol_ref.solution atol=1e-6
        @test sol_k25.constraints ≈ sol_ref.constraints atol=1e-6
        @test sol_k25.multipliers ≈ sol_ref.multipliers atol=1e-6
    end

    @testset "Regularization $(reg)" for reg in [
        MadQP.FixedRegularization(1e-8, -1e-9),
        MadQP.AdaptiveRegularization(1e-8, -1e-9, 1e-9),
    ]
        solver = MadQP.MPCSolver(
            qp;
            linear_solver=LDLSolver,
            print_level=MadNLP.ERROR,
            regularization=reg,
        )
        sol = MadQP.solve!(solver)

        @test sol.status == MadNLP.SOLVE_SUCCEEDED
        @test sol.objective ≈ sol_ref.objective atol=1e-6
        @test sol.solution ≈ sol_ref.solution atol=1e-6
        @test sol.constraints ≈ sol_ref.constraints atol=1e-6
        @test sol.multipliers ≈ sol_ref.multipliers atol=1e-6
    end
end

