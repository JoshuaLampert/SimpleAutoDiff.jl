module SimpleAutoDiff

include("dual_numbers.jl")
include("derivative.jl")
export DualNumber, value, derivative, gradient

end
