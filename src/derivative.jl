"""
    derivative(f, x0)

Compute the derivative of a function `f` mapping a scalar to a scalar evaluated at a value `x0`.
"""
derivative(f, x0) = derivative(f(DualNumber(x0, one(x0))))

"""
    derivative(f)

Compute the derivative of a function `f` and return the derivative as a function.
"""
derivative(f) = x0 -> derivative(f, x0)

"""
    gradient!(grad, f, x0)

Computes the gradient of a function `f` mapping a vector to a scalar evaluated at a value `x0`
in-place and overwrites `grad` by the gradient.

See also [`gradient`](@ref) for an out-of-place version of the function.
"""
function gradient!(grad, f, x0)
    x = similar(x0, DualNumber{eltype(x0)})
    copyto!(x, x0)
    for i in eachindex(x)
        x[i] = DualNumber(x0[i], one(eltype(x0)))
        grad[i] = derivative(f(x))
        x[i] = x0[i]
    end
    return grad
end

"""
    gradient(f, x0)

Compute the gradient of a function `f` mapping a vector to a scalar evaluated at a value `x0`.

See also [`gradient!`](@ref) for an in-place version of the function.
"""
function gradient(f, x0)
    grad = zeros(typeof(f(x0)), length(x0))
    return gradient!(grad, f, x0)
end

"""
    gradient(f)

Compute the gradient of a function `f` and return it as a function.f
"""
gradient(f) = x0 -> gradient(f, x0)
