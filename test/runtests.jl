using SimpleAutoDiff
using SpecialFunctions
using LinearAlgebra
using Test

@testset "SimpleAutoDiff.jl" begin
    @testset "Code quality" begin
        import Aqua
        using ExplicitImports: check_no_implicit_imports, check_no_stale_explicit_imports
        Aqua.test_all(ambiguities = false, SimpleAutoDiff)
        @test isnothing(check_no_implicit_imports(SimpleAutoDiff))
        @test isnothing(check_no_stale_explicit_imports(SimpleAutoDiff))
    end

    @testset "Dual numbers" begin
        x = DualNumber(1, 2)
        @test eltype(x) == Int64
        @test value(x) == 1
        @test derivative(x) == 2
        y = DualNumber(3, 4.0)
        @test eltype(y) == Float64
        @test eltype(typeof(y)) == Float64
        @test float(y) == y
        @test value(y) == 3.0
        @test derivative(y) == 4.0

        @test x + y == DualNumber(4.0, 6.0)
        @test x - y == DualNumber(-2.0, -2.0)
        @test x * y == DualNumber(3.0, 10.0)
        @test x / y == DualNumber(1 / 3, 2 / 9)
        @test x^y == DualNumber(1.0, 6.0)

        @test 1.0 + x == DualNumber(2.0, 2.0)

        @test !isinf(x)
        @test x < y
        @test x <= y
        @test y > 2
        @test 4 > y
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

        h = derivative(f2)
        @test h(2.0) == -0.25
        @test derivative(derivative(f1), 1.0) == 8.0
    end

    @testset "gradient" begin
        f1(x) = sin(x[1]) * cos(x[2])
        a = [pi, -pi]
        G = [1.0, 0.0]
        @test isapprox(gradient(f1, a), G, atol = 1e-15)
        @test gradient(x -> norm(x)^2, [1.0, 2.0]) == [2.0, 4.0]
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
        f2(x) = tanh(x[1]) * tan(x[1])
        hess = zeros(2, 2)
        x0 = [2.0, -1.0]
        @test_nowarn hessian!(hess, f2, x0)
        @test hess == hessian(f2, x0)

        h = hessian(f1)
        @test isapprox(h(a), H, atol = 1e-15)
    end
end
