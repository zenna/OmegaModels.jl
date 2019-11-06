module Fairness

using Omega
export gaussian, Hole, event, steps

include("helpers.jl")
include("evaluation.jl")
include("data.jl")

# Benchmarks
include("bench/therm_u10_b2.jl")
include("bench/dt16.jl")


end
