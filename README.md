# SimpleAutoDiff.jl

[![Build Status](https://github.com/JoshuaLampert/SimpleAutoDiff.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JoshuaLampert/SimpleAutoDiff.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/JoshuaLampert/SimpleAutoDiff.jl/graph/badge.svg?token=yKB7uIDHXE)](https://codecov.io/gh/JoshuaLampert/SimpleAutoDiff.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

**SimpleAutoDiff.jl** is [Julia](https://julialang.org/) package, which implements simple forward-mode automatic differentiation (AD). It is not
meant to be the most performant implementation, but rather focuses on simplicity and readability of the code. It only implements the core features
of AD without claiming optimal performance and flexibility. Therefore, it is a good starting point to understand or teach automatic/algorithmic
differentiation, but is not the perfect choice for production use. For alternative packages performing AD, see the list in https://juliadiff.org/.
The implementation in this package is closest to the implementation of [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl), but is a lot
simplified.

## Installation

If you have not yet installed Julia, then you first need to [download Julia](https://julialang.org/downloads/).
Please [follow the instructions for your operating system](https://julialang.org/downloads/platform/).
SimpleAutoDiff.jl works with Julia v1.10 and newer. You can install SimpleAutoDiff.jl by executing the following commands from the Julia REPL

```julia
julia> using Pkg

julia> Pkg.add("https://github.com/JoshuaLampert/SimpleAutoDiff.jl")
```

## Usage
In the Julia REPL, first load the package SimpleAutoDiff.jl

```julia
julia> using SimpleAutoDiff
```

Until now, SimpleAutoDiff.jl can only differentiate scalar functions of one or several variables, i.e. it can compute `derivative`s (scalar input)
and `gradient`s (vector input). To compute the derivative of a function $f: \mathbb{R}\to\mathbb{R}$, you can run

```julia
julia> f(x) = sin(x) * cos(x) + x^2
f (generic function with 1 method)

julia> derivative(f, pi)
7.283185307179586 # = 1 + 2pi
```

You can also get the `derivative` as a function

```julia
julia> g = derivative(f)
#1 (generic function with 1 method)

julia> g(pi)
7.283185307179586
```

Similarly, you can call `gradient` with a vector input.

```julia
julia> h(x) = cos(x[1]) * sin(x[2])
h (generic function with 1 method)

julia> gradient(h, [pi, 1])
2-element Vector{Float64}:
 -1.0305047481203616e-16
 -0.5403023058681398

julia> grad_h = gradient(h)
#3 (generic function with 1 method)

julia> grad_h([pi, 1])
2-element Vector{Float64}:
 -1.0305047481203616e-16
 -0.5403023058681398
 ```

## Authors

The package is developed and maintained by Joshua Lampert (University of Hamburg).

## License and contributing

SimpleAutoDiff.jl is published under the MIT license (see [License](https://github.com/JoshuaLampert/SimpleAutoDiff.jl/blob/main/LICENSE)).
We are pleased to accept contributions from everyone, preferably in the form of a PR.
