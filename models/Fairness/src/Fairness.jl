module Fairness

using Lens
using Omega
using ProgressMeter
using DataFrames
export gaussian, Hole, event, steps

include("helpers.jl")
include("evaluation.jl")
include("data.jl")
include("train.jl")

# Benchmarks
include("bench/therm_u10_b2.jl")
include("bench/dt16.jl")


end
