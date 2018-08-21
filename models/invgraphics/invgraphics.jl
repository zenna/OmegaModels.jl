"Inverse Graphics"
module InverseGraphics

using Omega
using RayTrace
using OmegaModels
using JLD2
using FileIO
using Flux

include("net.jl")
include("objects.jl")
include("prior.jl")
include("hyper.jl")

end