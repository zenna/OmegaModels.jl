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
using ..Train: train, TrainLoop, BatchLoop

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
  φ.simname = "infer"
  φ.name = "neuralscene"
  φ.runname = randrunname()
  φ.tags = ["test", "neuralscene"]
  φ.logdir = logdir(runname = φ.runname, tags = φ.tags)
  φ.runfile = joinpath(dirname(@__FILE__), "runscript.jl")
  # φ.gitinfo = RunTools.gitinfo()
  φ
end

"Optimization Parameters"
function optparams()
  Params((η = uniform([0.1, 0.01, 0.001, 0.0001]),
          opt = uniform([Descent, ADAM])))
end

function netparams()
  Params(midlen = uniform(40:60))
end

"All parameters"
function allparams()
  φ = Params()
  φ.scenelen = uniform(10:100)
  φ.niterations = uniform([1000, 2000, 5000, 10000, 50000])
  φ.imagesperbatch = uniform(1:10)
  merge(φ, runparams(), optparams(), netparams())
end

# Callbacks

# function saveimg(img, path)
#   backupsave()
# end

# function lmap()
#   saveimgs = incsave(joinpath(joinpath(φ.logdir, "raytraced.jld2"))) ∘ (x -> x.neural_img)
#   stopnanorinf
#   # Show/save images every n iterations
#   everyn
# end

function infer(φ)
  display(φ)

  # BatchLoop callbacks

  # Save weights
  # savew = savecb(joinpath(φ.logdir, "raytraced.jld2"); backup = true, verbose = true)  ∘ (x -> Dict("weights" => x.neural_img))

  # Save image
  simg = incsave(joinpath(φ.logdir, "raytraced.jld2"); verbose = true) ∘ (x -> Dict("x" => x.neural_img))

  # TrainLoop Callbacks
  sp = showprogress(φ.niterations)
  sl = Callbacks.plotscalar() ∘ (x -> (x = x.i, y = x.loss))
  x = 2
  lmap = (BatchLoop => everyn(simg, 50), TrainLoop => runall([sp, sl])) 

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
  @leval lmap train(; net = net_,
                      deepscene = deepscene,
                      opt = φ.opt(φ.η),
                      niterations = φ.niterations,
                      imagesperbatch = φ.imagesperbatch,
                      render_params = render_params)
end

function testhyper()
  p = rand(allparams())
  mkpath(p.logdir)
  infer(p)
end



end