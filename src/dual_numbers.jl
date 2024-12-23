"""
    DualNumber(value, deriv)

A struct representing a dual number ``a + εb``, where ``a`` and ``b`` are real numbers
representing the `value` and the derivative `deriv` of a function in the context of automatic
differentiation. Here, ``ε`` is a symbol satisfying ``ε^2 = 0``.
"""
struct DualNumber{T <: Real} <: Number
    value::T
    deriv::T
end

DualNumber(value::Real, deriv::Real) = DualNumber(promote(value, deriv)...)

value(x::DualNumber) = x.value
derivative(x::DualNumber) = x.deriv

Base.real(::DualNumber{T}) where {T} = T

Base.:+(x::DualNumber, y::DualNumber) = DualNumber(x.value + y.value, x.deriv + y.deriv)
Base.:-(x::DualNumber, y::DualNumber) = DualNumber(x.value - y.value, x.deriv - y.deriv)
function Base.:*(x::DualNumber, y::DualNumber)
    DualNumber(x.value * y.value, x.value * y.deriv + x.deriv * y.value)
end
function Base.:/(x::DualNumber, y::DualNumber)
    DualNumber(x.value / y.value, (x.deriv * y.value - x.value * y.deriv) / y.value^2)
end

Base.convert(::Type{DualNumber{T}}, x::Real) where {T <: Real} = DualNumber(x, zero(T))
Base.promote_rule(::Type{DualNumber{T}}, ::Type{<:Real}) where {T <: Real} = DualNumber{T}
