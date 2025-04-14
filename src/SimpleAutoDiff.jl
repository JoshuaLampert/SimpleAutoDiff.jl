module SimpleAutoDiff

import DiffRules
import SpecialFunctions
import LogExpFunctions
import NaNMath

include("dual_numbers.jl")
include("derivative.jl")
export DualNumber,
       value, derivative, gradient, gradient!, jacobian, jacobian!, hessian, hessian!

end
