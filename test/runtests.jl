using SimpleAutoDiff
using Test

@testset "SimpleAutoDiff.jl" begin
    @testset "Dual numbers" begin
        x = DualNumber(1, 2)
        @test real(x) == Int64
        @test value(x) == 1
        @test derivative(x) == 2
        y = DualNumber(3, 4.0)
        @test real(y) == Float64
        @test value(y) == 3.0
        @test derivative(y) == 4.0

        @test x + y == DualNumber(4.0, 6.0)
        @test x - y == DualNumber(-2.0, -2.0)
        @test x * y == DualNumber(3.0, 10.0)
        @test x / y == DualNumber(1/3, 2/9)

        @test 1.0 + x == DualNumber(2.0, 2.0)
    end

    @testset "derivative" begin
        f(x) = 4.0 * x^2 - 1.0
        @test derivative(f, 3.0) == 24.0
        g(x) = 1/x
        @test derivative(g, 2) == -0.25
    end
end
