module NeuralScene

using RayTrace
using Flux
using Zygote
import DSLearn
using Parameters
using LinearAlgebra: norm
# using UnicodePlots
using RayTrace: Ray, Scene, sceneintersect, trcdepth
using GeometryTypes
using Lens
using ZenUtils

export train, TrainLoop

include("gendata.jl")   # Generate synthetic data
using .GenData

include("linearize.jl") # linearization of data structures
using .Linearize

include("model.jl")     # Model
using .Model

include("loss.jl")      # Loss

include("interactive.jl") # Interactive plotting

include("distances.jl")
using .Distances

include("train.jl")
using .Train

include("zygote.jl")

# include("main.jl")

include("run.jl")
using .Run

end