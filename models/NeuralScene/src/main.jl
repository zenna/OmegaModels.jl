
# Create a random neural scene
using NeuralScene
using NeuralScene.Model: DeepScene
using NeuralScene.Linearize: linearlength
using NeuralScene.GenData: ex_data
using Zygote
using GeometryTypes
using Flux
using RayTrace: sceneintersect, Ray
import RayTrace
using DSLearn

deepscene = DeepScene(rand(10))
inlen = linearlength(Vector{Float64}, (Ray{Point3, Point3}, deepscene))
# outlen = linearlength(Vector{Float64}, (Bool, Float64, Float64))
outlen = 1
# trackednet = Flux.Dense(inlen, outlen)
midlen = 50
trackednet = Flux.Chain(Flux.Dense(inlen, midlen),
                              Flux.Dense(midlen, outlen))


# Until Flux drops Tracker as its default Automatic Differentiation library,
# strip it out with this line:
net_ = Flux.mapleaves(Flux.data, trackednet)

DSLearn.net(::Type{typeof(sceneintersect)}, ::Type{Ray}, ::Type{DeepScene}) = net_

# Show rendered example scene
x = ex_data()
img = x.img

# Render the scene to get an (untrained) image
neural_img = RayTrace.renderfunc(deepscene; x.render_params...)

# # Train the network
params = [deepscene.ir]

# # Compute gradients
function f()
  g = gradient(Params(params)) do
    sum(RayTrace.renderfunc(deepscene; x.render_params...))
  end
end