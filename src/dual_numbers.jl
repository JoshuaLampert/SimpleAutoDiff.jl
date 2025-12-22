"""
    DualNumber(value, deriv)

A struct representing a dual number ``a + εb``, where ``a`` and ``b`` are real numbers
representing the `value` and the derivative `deriv` of a function in the context of automatic
differentiation. Here, ``ε`` is a symbol satisfying ``ε^2 = 0``.
"""
struct DualNumber{T <: Real} <: Real
    value::T
    deriv::T
end

function DualNumber(value::T1, deriv::T2) where {T1, T2}
    T = promote_type(T1, T2)
    return DualNumber(convert(T, value), convert(T, deriv))
end

value(d::DualNumber) = d.value
derivative(d::DualNumber) = d.deriv

Base.eltype(::DualNumber{T}) where {T} = T
Base.eltype(::Type{DualNumber{T}}) where {T} = T
Base.float(::Type{DualNumber{T}}) where {T} = DualNumber{T}
Base.float(d::DualNumber) = convert(float(typeof(d)), d)

Base.convert(::Type{DualNumber{T}}, x::Number) where {T} = DualNumber(x, zero(T))
Base.convert(::Type{D}, d::D) where {D <: DualNumber} = d
Base.promote_rule(::Type{DualNumber{T}}, ::Type{<:Real}) where {T <: Real} = DualNumber{T}

# diff rules
# Define binary diff rules by hand. We are missing some from DiffRules.jl (e.g. atan, hypot, ...),
# but since they are not essential we skip them for simplicity.
Base.:+(x::DualNumber, y::DualNumber) = DualNumber(x.value + y.value, x.deriv + y.deriv)
Base.:-(x::DualNumber, y::DualNumber) = DualNumber(x.value - y.value, x.deriv - y.deriv)
function Base.:*(x::DualNumber, y::DualNumber)
    return DualNumber(x.value * y.value, x.value * y.deriv + x.deriv * y.value)
end
function Base.:/(x::DualNumber, y::DualNumber)
    return DualNumber(x.value / y.value,
                      (x.deriv * y.value - x.value * y.deriv) / y.value^2)
end
function Base.:^(x::DualNumber, y::DualNumber)
    return DualNumber(x.value^y.value,
                      x.value^(y.value - 1) *
                      (y.value * x.deriv + x.value * y.deriv * log(x.value)))
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

const UNARY_PREDICATES = Symbol[:isinf, :isnan, :isfinite, :iseven, :isodd, :isreal,
                                :isinteger]

for pred in UNARY_PREDICATES
    @eval Base.$(pred)(d::DualNumber) = $(pred)(value(d))
end

const BINARY_PREDICATES = Symbol[:isequal, :isless, :<, :>, :(==), :(!=), :(<=), :(>=)]
# The ambiguous types are needed to avoid method ambiguities. We need to specialize the second non-DualNumber argument.
# From https://github.com/JuliaDiff/ForwardDiff.jl/blob/v0.10.38/src/dual.jl#L146 and
# https://github.com/JuliaDiff/ForwardDiff.jl/blob/v0.10.38/src/prelude.jl#L9
const AMBIGUOUS_TYPES = (AbstractFloat, Irrational, Integer, Rational, Real, RoundingMode)

for pred in BINARY_PREDICATES
    @eval Base.$(pred)(x::DualNumber, y::DualNumber) = $(pred)(value(x), value(y))
    for R in AMBIGUOUS_TYPES
        @eval Base.$(pred)(x::DualNumber, y::$R) = $(pred)(value(x), y)
        @eval Base.$(pred)(x::$R, y::DualNumber) = $(pred)(x, value(y))
    end
end
