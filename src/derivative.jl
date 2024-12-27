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

Compute the gradient of a function `f` and return it as a function.
"""
gradient(f) = x0 -> gradient(f, x0)

"""
    jacobian!(jac, f, x0)

Computes the Jacobian matrix of a function `f` mapping a vector to a vector evaluated at a value `x0`
in-place and overwrites `jac` by the Jacobian.

See also [`jacobian`](@ref) for an out-of-place version of the function.
"""
function jacobian!(jac, f, x0)
    x = similar(x0, DualNumber{eltype(x0)})
    copyto!(x, x0)
    for i in eachindex(x)
        x[i] = DualNumber(x0[i], one(eltype(x0)))
        fx = f(x)
        for j in eachindex(fx)
            jac[j, i] = derivative(fx[j])
        end
        x[i] = x0[i]
    end
    return jac
end

"""
    jacobian(f, x0)

Compute the Jacobian matrix of a function `f` mapping a vector to a vector evaluated at a value `x0`.

See also [`jacobian!`](@ref) for an in-place version of the function.
"""
function jacobian(f, x0)
    fx0 = f(x0)
    jac = zeros(eltype(fx0), length(fx0), length(x0))
    return jacobian!(jac, f, x0)
end

"""
    jacobian(f)

Compute the Jacobian matrix of a function `f` and return it as a function.
"""
jacobian(f) = x0 -> jacobian(f, x0)
