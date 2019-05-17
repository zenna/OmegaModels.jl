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

# TODO:
# Periodically save parameter weights with backup
# Make plot to tensorbard

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
  φ.runname = ciid(randrunname)
  φ.tags = ["test", "neuralscene"]
  φ.logdir = logdir(runname = φ.runname, tags = φ.tags)
  φ.runfile = joinpath(dirname(@__FILE__), "..", "scripts", "runscript.jl")
  φ.gitinfo = current_commit(@__FILE__)
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
  φ = Params(scenelen = uniform(10:100),
             width = 100,
             height = 100,
             niterations = uniform([1000, 2000, 5000, 10000, 50000]),
             normalizeimgs = bernoulli(0.5, Bool),
             imagesperbatch = uniform(1:10))
  merge(φ, runparams(), optparams(), netparams())
end

"Generate a Lens map"
function genlmap(φ)
  simg = incsave(joinpath(φ.logdir, "raytraced.jld2"); verbose = true) ∘ (x -> Dict("x" => x.neural_img))
  sp = showprogress(φ.niterations)
  sl = Callbacks.plotscalar() ∘ (x -> (x = x.i, y = x.loss))
  lmap = (BatchLoop => everyn(simg, 50), TrainLoop => runall([sp, sl])) 
end

"Generate the network from parameter values"
function gennet(φ, deepscene)
  inlen = linearlength(Vector{Float64}, (Ray{Point3, Point3}, deepscene))
  midlen = φ.midlen
  outlen = 1
  trackednet = Flux.Chain(Flux.Dense(inlen, midlen),
                          Flux.Dense(midlen, outlen))
  Flux.mapleaves(Flux.data, trackednet)
end

function infer(φ)
  display(φ)

  # Setup scene
  normalize = φ.normalizeimgs ? identity : x -> x ./ sum(x)
  deepscene = DeepScene(rand(φ.scenelen))

  # Network
  net_ = gennet(φ, deepscene)
  net.net = net_  # Set's the global net

  # TrainLoop Callbacks
  lmap = genlmap(φ)
  
  render_params = (width = φ.width, height = φ.height, fov = 30.0, trc = trcdepth)
  @leval lmap train(; net = net_,
                      deepscene = deepscene,
                      opt = φ.opt(φ.η),
                      niterations = φ.niterations,
                      imagesperbatch = φ.imagesperbatch,
                      render_params = render_params,
                      normalize = normalize)
end

function testhyper()
  p = rand(allparams())
  mkpath(p.logdir)
  infer(p)
end

end