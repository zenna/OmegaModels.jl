"Run a bunch of simulations"
module Run
using RunTools
using Lens, Callbacks
using Flux
using ..Model: DeepScene
using ..Viz: unicodeplotmat
using ..Linearize
using GeometryTypes
using RayTrace: Ray, sceneintersect, trcdepth
using Omega
import DSLearn
using ..Train: train, TrainLoop, BatchLoop
using BSON

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
  φ.tags = ["neuralscene", "first"]
  φ.logdir = ciid(ω -> logdir(runname = φ.runname(ω), tags = φ.tags))
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
            #  niterations = 20,
             normalizeimgs = bernoulli(0.5, Bool),
             imagesperbatch = uniform(1:10))
  merge(φ, runparams(), optparams(), netparams())
end

"Generate a Lens map"
function genlmap(φ)
  # save network opt scene
  savenet = incsave(joinpath(φ.logdir, "net.bson"); verbose = true,
                    save = BSON.bson) ∘ (x -> Dict(:net => x.net, :deepscene => x.deepscene))
  
  # Plot image
  simg_ = incsave(joinpath(φ.logdir, "raytraced.jld2"); verbose = true) ∘ (x -> Dict("x" => x.neural_img))
  simg = everyn(simg_, 50)

  # UnicodePlot Nueral Scene
  plotscene = everyn(unicodeplotmat ∘ (x -> x.neural_img), 10 * φ.imagesperbatch)

  # Stopping
  stop = stopnanorinf ∘ (x -> x.loss)

  # Stop when converged
  stopconv = everyn(stopconverged(; verbose = true) ∘ (x -> (x.i, x.loss)), 50)

  # Show Progress
  sp = showprogress(φ.niterations)

  # Plos los
  sl = Callbacks.plotscalar(; width = 100, height = 30) ∘ (x -> (x = x.i, y = x.loss))

  # The lensmap
  lmap = (BatchLoop => runall(simg, stop, plotscene),
          TrainLoop => runall(sp, sl, everyn(savenet, div(φ.niterations, 10)), stopconv))
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