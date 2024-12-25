"""
    DualNumber(value, deriv)

A struct representing a dual number ``a + εb``, where ``a`` and ``b`` are real numbers
representing the `value` and the derivative `deriv` of a function in the context of automatic
differentiation. Here, ``ε`` is a symbol satisfying ``ε^2 = 0``.
"""
struct DualNumber{T<:Real} <: Number
    value::T
    deriv::T
end

DualNumber(value::Real, deriv::Real) = DualNumber(promote(value, deriv)...)

value(d::DualNumber) = d.value
derivative(d::DualNumber) = d.deriv

Base.real(::DualNumber{T}) where {T} = T
Base.float(::Type{DualNumber{T}}) where {T} = DualNumber{T}
Base.float(d::DualNumber) = convert(float(typeof(d)), d)

Base.convert(::Type{DualNumber{T}}, x::Real) where {T<:Real} = DualNumber(x, zero(T))
Base.promote_rule(::Type{DualNumber{T}}, ::Type{<:Real}) where {T<:Real} = DualNumber{T}

# rules
Base.:+(x::DualNumber, y::DualNumber) = DualNumber(x.value + y.value, x.deriv + y.deriv)
Base.:-(x::DualNumber, y::DualNumber) = DualNumber(x.value - y.value, x.deriv - y.deriv)
function Base.:*(x::DualNumber, y::DualNumber)
    DualNumber(x.value * y.value, x.value * y.deriv + x.deriv * y.value)
end
function Base.:/(x::DualNumber, y::DualNumber)
    DualNumber(x.value / y.value, (x.deriv * y.value - x.value * y.deriv) / y.value^2)
end

# For each unitary rule in DiffRules.jl define a function dispatching on `DualNumber`
for (M, f, arity) in DiffRules.diffrules(filter_modules = nothing)
    if arity == 1
        Mf = M == :Base ? f : :($M.$f)
        @eval function $M.$f(d::DualNumber)
            x = $SimpleAutoDiff.value(d)
            val = $Mf(x)
            deriv = $(DiffRules.diffrule(M, f, :x))
            return DualNumber(val, deriv * derivative(d))
        end
    end
end

const UNARY_PREDICATES =
    Symbol[:isinf, :isnan, :isfinite, :iseven, :isodd, :isreal, :isinteger]

for pred in UNARY_PREDICATES
    @eval Base.$(pred)(d::DualNumber) = $(pred)(value(d))
end

const BINARY_PREDICATES = Symbol[:isequal, :isless, :<, :>, :(==), :(!=), :(<=), :(>=)]

for pred in BINARY_PREDICATES
    @eval Base.$(pred)(x::DualNumber, y::DualNumber) = $(pred)(value(x), value(y))
    @eval Base.$(pred)(x::DualNumber, y) = $(pred)(value(x), y)
    @eval Base.$(pred)(x, y::DualNumber) = $(pred)(x, value(y))
end
