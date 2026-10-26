using SimpleAutoDiff
using SpecialFunctions
using LinearAlgebra
using Test

@testset "SimpleAutoDiff.jl" begin
    @testset "Code quality" begin
        import Aqua
        using ExplicitImports: test_explicit_imports
        Aqua.test_all(SimpleAutoDiff)
        # `diffrules` and `diffrule` are documented API of DiffRules.jl, but not declared `public`
        ignore = (:diffrules, :diffrule)
        test_explicit_imports(SimpleAutoDiff;
                              all_qualified_accesses_are_public = (; ignore))
    end

    @testset "Dual numbers" begin
        x = DualNumber(1, 2)
        @test x isa DualNumber{Int64}
        @test value(x) == 1
        @test derivative(x) == 2
        y = DualNumber(3, 4.0)
        @test y isa DualNumber{Float64}
        @test float(y) === y
        @test value(y) == 3.0
        @test derivative(y) == 4.0

        # `==` only compares the values of dual numbers. Use `===` to also compare
        # the derivatives (and the types).
        @test DualNumber(1.0, 2.0) == DualNumber(1.0, 3.0)
        @test DualNumber(1.0, 2.0) !== DualNumber(1.0, 3.0)
        @test x + y === DualNumber(4.0, 6.0)
        @test x - y === DualNumber(-2.0, -2.0)
        @test x * y === DualNumber(3.0, 10.0)
        @test x / y === DualNumber(1 / 3, 2 / 9)
        @test x^y === DualNumber(1.0, 6.0)

        @test 1.0 + x === DualNumber(2.0, 2.0)

        @test !isinf(x)
        @test x < y
        @test x <= y
        @test y > 2
        @test 4 > y
        @test x != pi

        # `hash` must be consistent with `isequal`, which only compares the values
        @test isequal(DualNumber(1.0, 2.0), DualNumber(1.0, 3.0))
        @test hash(DualNumber(1.0, 2.0)) == hash(DualNumber(1.0, 3.0))
        @test hash(DualNumber(1.0, 2.0)) == hash(1.0)
        @test haskey(Dict(DualNumber(1.0, 2.0) => 1), 1.0)
        duals = [DualNumber(1.0, 2.0), DualNumber(1.0, 3.0), DualNumber(2.0, 2.0)]
        @test length(Set(duals)) == 2

        # Like other numbers, dual numbers behave as scalars in iteration and broadcasting
        @test eltype(y) === typeof(y)
        @test only(collect(y)) === y
        @test all(y * [1.0, 2.0] .=== [DualNumber(3.0, 4.0), DualNumber(6.0, 8.0)])

        # Conversion and promotion
        @test_throws MethodError DualNumber(1 + 2im, 1)
        @test float(x) === DualNumber(1.0, 2.0)
        @test convert(DualNumber{Float32}, 0.1) === DualNumber(0.1f0, 0.0f0)
        @test_throws InexactError convert(DualNumber{Int}, 0.5)
        @test convert(DualNumber{Float64}, x) === DualNumber(1.0, 2.0)
        @test promote_type(DualNumber{Float32}, Float64) == DualNumber{Float64}
        @test promote_type(DualNumber{Int}, DualNumber{Float64}) == DualNumber{Float64}
        @test [x, y] isa Vector{DualNumber{Float64}}
        # A dual number with smaller nesting depth is constant w.r.t. the outer perturbation
        z = DualNumber(y, DualNumber(1.0, 0.0))
        @test convert(typeof(z), y) === DualNumber(y, DualNumber(0.0, 0.0))
        @test promote_type(typeof(z), DualNumber{Float32}) == typeof(z)
        @test y * z === DualNumber(y * y, y)
    end

    @testset "derivative" begin
        f1(x) = 4.0 * x^2 - 1.0
        @test derivative(f1, 3.0) == 24.0
        f2(x) = 1 / x
        @test derivative(f2, 2) == -0.25
        f3(x) = sin(x) * cos(x)
        @test derivative(f3, pi) == 1.0
        f4(x) = log(x) * exp(x)
        a = 42.1
        @test isapprox(derivative(f4, a), exp(a) * (a * log(a) + 1) / a, atol = 1e-14)
        f5(x) = tan(x)
        @test isapprox(derivative(f5, a), sec(a)^2, atol = 1e-14)
        f6(x) = x * abs(x)
        @test derivative(f6, 2.0) == 4.0
        f7(x) = gamma(x)
        @test derivative(f7, a) == digamma(a) * gamma(a)
        f8(x) = 2^x
        @test derivative(f8, a) == log(2) * 2^a
        f9(x) = sum(x * [1.0, 2.0])
        @test derivative(f9, a) == 3.0
        f10(x) = x^2.0
        @test derivative(f10, -2.0) == -4.0
        @test derivative(f10, 0.0) == 0.0

        h = derivative(f2)
        @test h(2.0) == -0.25
        @test derivative(derivative(f1), 1.0) == 8.0
        # Nested derivatives of closures with different nesting depth
        f10(x) = x * derivative(y -> x * y, x)
        @test derivative(f10, 3.0) == 6.0
        # Perturbation confusion: both perturbations have the same nesting depth, which
        # cannot be distinguished without tags (as in ForwardDiff.jl)
        f11(x) = x * derivative(y -> x + y, 1.0)
        @test_broken derivative(f11, 1.0) == 1.0
    end

    @testset "gradient" begin
        f1(x) = sin(x[1]) * cos(x[2])
        a = [pi, -pi]
        G = [1.0, 0.0]
        @test isapprox(gradient(f1, a), G, atol = 1e-15)
        @test gradient(x -> norm(x)^2, [1.0, 2.0]) == [2.0, 4.0]
        @test gradient(norm, [3, 4]) ≈ [0.6, 0.8]
        f2(x) = tanh(x[1]) * tan(x[1])
        grad = zeros(2)
        x0 = [2.0, -1.0]
        @test_nowarn gradient!(grad, f2, x0)
        @test grad == gradient(f2, x0)

        h = gradient(f1)
        @test isapprox(h(a), G, atol = 1e-15)
    end

    @testset "jacobian" begin
        f1(x) = [sin(x[1]) * cos(x[2]), tanh(x[2] * sec(x[1])), acos(x[1] + x[2])]
        a = [pi, -pi]
        J = [1.0 0.0
             0.0 -0.007441950142796139
             -1.0 -1.0]
        @test isapprox(jacobian(f1, a), J, atol = 1e-15)
        f2(x) = [tanh(x[3] * x[2]^2) * exp(x[1]), log(x[3] * x[1])]
        jac = zeros(2, 3)
        x0 = [2.0, -1.0, pi]
        @test_nowarn jacobian!(jac, f2, x0)
        @test jac == jacobian(f2, x0)
        f3(x) = x[1] * [1.0, 2.0]
        @test jacobian(f3, [3.0]) == reshape([1.0, 2.0], 2, 1)

        h = jacobian(f1)
        @test isapprox(h(a), J, atol = 1e-15)
    end

    @testset "hessian" begin
        f1(x) = sin(x[1]^2) * cos(x[2]^2)
        a = [pi, -pi]
        H = [-13.704786185347317 15.334467910643793
             15.334467910643793 -15.704786185347313]
        @test isapprox(hessian(f1, a), H, atol = 1e-15)
        @test hessian(x -> norm(x)^2, [1.0, 2.0]) == [2.0 0.0
                                                      0.0 2.0]
        @test hessian(x -> norm(x)^2, [1, 2]) ≈ [2.0 0.0
                                                 0.0 2.0]
        f2(x) = tanh(x[1]) * tan(x[1])
        hess = zeros(2, 2)
        x0 = [2.0, -1.0]
        @test_nowarn hessian!(hess, f2, x0)
        @test hess == hessian(f2, x0)

        h = hessian(f1)
        @test isapprox(h(a), H, atol = 1e-15)
    end
end
