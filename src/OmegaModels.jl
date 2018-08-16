__precompile__(false)
"Collection of Models"
module OmegaModels

using Omega

include("mnist/mnist.jl")
include("thermostat/thermostat.jl")
include("invgraphics/invgraphics.jl")
include("programlearn/programlearn.jl")

end