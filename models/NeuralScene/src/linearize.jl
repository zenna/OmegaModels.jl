"Linearize data structures"
module Linearize

using StaticArrays

export linearize, linearsize, linearlength, unlinearize

linearize(::Type{T}, x::T) where T = x

# zt: abstract vector?
linearize(::Type{Vector{T}}, x::Vector) where T = x
linearize(::Type{Vector{T}}, x::NTuple{N, T}) where {T, N} = T[x...]
linearize(::Type{Vector{T}}, x::T) where T = T[x]

combine(::Type{Vector{T}}, xs::Vector{Vector{T}}) where T = vcat(xs...)

"Get fields of `x` as a vector"
fields(x::T) where T = map(k -> getfield(x, k), 1:length(fieldnames(T)))
fields(x::Type{T}) where T= fieldtypes(T)

"""
`linearize(T, x)`

Linearize a data structure into a collection of type T, typically in-order for
it to be used as input to a procedure (e.g. a neural network) that accepts only
linear input.

```julia
struct MyType
  x::Vector
  y::Tuple
  z::Float64
end

x = MyType([1.0, 2.0], (1.0, 2.0), 3.0)
linearize(Vector{Float64}, x)
"""
function linearize(T::Type{ColT}, x::XT) where {ColT, XT}
  # zt: this will match things which are not complex
  xs = map(x_ -> linearize(T, x_), fields(x))  # xs
  combine(T, xs)
end

# Linear Size/Length
"""
`linearsize(T, x)`

Size of `x` linearized

`linearsize(T, x) = size(linearize(T, x))`
"""
function linearlength end
# linearlength(T, x) = @show length(linearize(T, x))
linearlength(T, x::Type{<:Real}) = 1
linearlength(T, x::Real) = 1
linearlength(T, x::Vector) = length(x)

# Make static
# linearlength(::Type{Vector{T}}, x::NTuple{N, T}) where {T, N} = N
# linearlength(::Type{Vector{T}}, x::NTuple{N, T}) where {T, N} = N
# linearlength(::Type{Vector{T}}, x::T) where T = 1
linearlength(::Type{Vector{T}}, ::Type{TT}) where {T,  TT <: StaticArray} = length(TT)

function linearlength(T, x)
  nfields(x) == 0 && error("need more than 0 fields") 
  xs = map(x_ -> linearlength(T, x_), fields(x))
  sum(xs)
end

# Should the length depend on the collection type?

# Unlinearization

"`unlinearize(T, x)` returns a composite structure of type `T` from linear `x`"
function unlinearize end

# Unlinearization is more complex
# 1. do i want to coerce a a float into a Bool?
# 2. How should we do it?

# function unlinearize(::Type{NamedTuple{K, T}}, xs) where {K, T}
#   NamedTuple{K}(map(unlinearize, T, xs))
# end

end