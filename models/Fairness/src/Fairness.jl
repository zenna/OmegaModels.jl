module Fairness

using Lens
using Omega
using Callbacks
using SuParameters
using RunTools
using ProgressMeter
using DataFrames
using BSON
export gaussian, Hole, event, steps


scope!("Fairness")

include("helpers.jl")
include("evaluation.jl")
include("data.jl")
include("train.jl")

# Benchmarks
include("bench/therm_u10_b2.jl")
include("bench/dt16.jl")


end
