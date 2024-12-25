using SimpleAutoDiff
using SpecialFunctions
using LinearAlgebra
using Test

@testset "SimpleAutoDiff.jl" begin
    @testset "Dual numbers" begin
        x = DualNumber(1, 2)
        @test real(x) == Int64
        @test value(x) == 1
        @test derivative(x) == 2
        y = DualNumber(3, 4.0)
        @test real(y) == Float64
        @test float(y) == y
        @test value(y) == 3.0
        @test derivative(y) == 4.0

        @test x + y == DualNumber(4.0, 6.0)
        @test x - y == DualNumber(-2.0, -2.0)
        @test x * y == DualNumber(3.0, 10.0)
        @test x / y == DualNumber(1 / 3, 2 / 9)

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

        h = derivative(f2)
        @test h(2.0) == -0.25
    end

    @testset "gradient" begin
        f1(x) = sin(x[1]) * cos(x[2])
        @test isapprox(gradient(f1, [pi, -pi]), [1.0, 0.0], atol = 1e-15)
        @test gradient(x -> norm(x)^2, [1.0, 2.0]) == [2.0, 4.0]

        h = gradient(f1)
        @test isapprox(h([-pi, pi]), [1.0, 0.0], atol = 1e-15)
    end
end
