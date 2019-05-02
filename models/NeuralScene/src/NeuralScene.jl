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

# TODO
# Implement linearsize
# Get params
# Delinearization

include("gendata.jl")   # Generate synthetic data
using .GenData

include("linearize.jl") # linearization of data structures
using .Linearize

include("model.jl")     # Model
using .Model

include("loss.jl")      # Loss

# Create a random neural scene
const deepscene = DeepScene(rand(10))
const inlen = linearlength(Vector{Float64}, (Ray{Point3, Point3}, deepscene))
# const outlen = linearlength(Vector{Float64}, (Bool, Float64, Float64))
const outlen = 1
const trackednet = Flux.Dense(inlen, outlen)

# Until Flux drops Tracker as its default Automatic Differentiation library,
# strip it out with this line:
const net_ = Flux.mapleaves(Flux.data, trackednet)

DSLearn.net(::Type{typeof(sceneintersect)}, ::Type{Ray}, ::Type{DeepScene}) = net_

# Show rendered example scene
x = ex_data()
const img = x.img

# Render the scene to get an (untrained) image
const neural_img = RayTrace.renderfunc(deepscene; x.render_params...)

# # Train the network
const params = [deepscene.ir]

distance(x, y) = norm(x - y)

fktrace(args...) = 0.3

# # Compute gradients
function f()
  g = gradient(Params(params)) do
    @show "hi"
    sum(RayTrace.renderfunc(deepscene; width = 4, height = 4, trc = faketrc))
    # distance(RayTrace.render(deepscene; render_params...), img)
  end
end 
# # Vizualise the gradients

# # Now do real training

end