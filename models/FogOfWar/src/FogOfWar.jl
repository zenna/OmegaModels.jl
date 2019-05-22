module FogOfWar


include("model.jl")
using .Model

include("viz.jl")
using .Viz


greet() = print("Hello World!")

end # module
