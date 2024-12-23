"""
    derivative(f, x)

Compute the derivative of a function `f` mapping a real number to a real number evaluated a value `x`.
"""
derivative(f, x) = derivative(f(DualNumber(x, one(x))))
