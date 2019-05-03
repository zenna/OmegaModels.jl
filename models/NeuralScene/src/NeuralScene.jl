module NeuralScene

# Zygote issues

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

include("train.jl")
using .Train

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

# # Compute gradients
function f()
  g = gradient(Params(params)) do
    sum(RayTrace.renderfunc(deepscene; x.render_params...))
  end
end 
# # Vizualise the gradients

# # Now do real training
function train(opt = ADAM(0.001), niterations = 100)
  W_ = net_.W
  params_ = (deepscene.ir, W_)
  paramsa = Params([deepscene.ir, W_])
  for i = 1:niterations
    grads = gradient(paramsa) do
      neural_img = RayTrace.renderfunc(deepscene; x.render_params...)
      loss = distance(neural_img, img)
    end
    grads_ = map(x -> grads[x], params_)
    zyg_update!(opt, params_, grads_)
  end
end

# @leval train()

end