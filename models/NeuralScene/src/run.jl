"Run a bunch of simulations"
module Run
using RunTools
using Lens, Callbacks
using Flux
using ..Model: DeepScene
using ..Linearize
using GeometryTypes
using RayTrace: Ray, sceneintersect, trcdepth
using Omega
import DSLearn
using ..Train: train

# Goals:
# Randomize length of inner vector
# Update net accordingly
# Also randomize net parameters
# Periodically render and save an image
# Periodically save parameter weights with backup
# Make plot to tensorbard
# How to update the net?
# 

mutable struct NetWrap
  net
end

const net = NetWrap(nothing)

DSLearn.net(::Type{typeof(sceneintersect)}, ::Type{Ray}, ::Type{DeepScene}) = net.net

function runparams()
  φ = Params()
  φ.train = true
  φ.loadnet = false
  φ.name = "neuralscene"
  φ.runname = randrunname()
  φ.tags = ["test", "neuralscene"]
  φ.logdir = logdir(runname = φ.runname, tags = φ.tags)
  φ.runfile = @__FILE__
  # φ.gitinfo = RunTools.gitinfo()
  φ
end

"Optimization Parameters"
function optparams()
  Params((η = uniform([0.1, 0.01, 0.001, 0.0001]),
          opt = uniform([Descent, ADAM])))
end

function netparams()
  Params((midlen = uniform(40:60),))
end

"All parameters"
function allparams()
  φ = Params()
  φ.scenelen = uniform(10:100)
  φ.niterations = uniform([1000, 2000, 5000, 10000, 50000])
  φ.imagesperbatch = uniform(1:10)
  merge(φ, runparams(), optparams(), netparams())
end

"Parameters we wish to enumerate"
function enumparams()
  [Params()]
end

function paramsamples(nsamples = 10)
  (rand(merge(allparams(), φ, Params(Dict(:samplen => i))))  for φ in enumparams(), i = 1:nsamples)
end

function infer(φ)
  display(φ)
  # Setup callbacks
  render_params = (width = 100, height = 100, fov = 30.0, trc = trcdepth)
  deepscene = DeepScene(rand(φ.scenelen))
  inlen = linearlength(Vector{Float64}, (Ray{Point3, Point3}, deepscene))
  midlen = φ.midlen
  outlen = 1
  trackednet = Flux.Chain(Flux.Dense(inlen, midlen),
                          Flux.Dense(midlen, outlen))
  net_ = Flux.mapleaves(Flux.data, trackednet)
  net.net = net_
  # net_ = 
  train(; net = net_,
                      deepscene = deepscene,
                      opt = φ.opt(φ.η),
                      niterations = φ.niterations,
                      imagesperbatch = φ.imagesperbatch,
                      render_params = render_params)
end

function testhyper()
  p = first(paramsamples())
  mkpath(p.logdir)
  infer(p)
end

main() = RunTools.control(infer, paramsamples())

end