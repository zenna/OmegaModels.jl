__precompile__(false)
"Inverse Graphics"
module InvRayTrace

using Omega
using RayTrace
using OmegaModels
using JLD2
using FileIO
using Flux
using RunTools
using DataFrames
using Tensorboard
import Statistics: mean

include("net.jl")
include("objects.jl")
include("prior.jl")
include("hyper.jl")

end