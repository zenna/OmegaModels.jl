module Distances
using LinearAlgebra
export distance

distance(x, y) = norm(x - y)

end