"Run a bunch of simulations"
module Run
using RunTools
import RayTrace
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
using TensorBoardLogger
using Logging
using ..GenData


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
  Params((η = uniform([0.01, 0.001, 0.0001]),
          opt = uniform([Descent, ADAM])))
end

function netparams()
  Params(midlen = uniform(40:60),
         nhidden = uniform(0:8),
         activation = uniform([relu, elu, selu]))
end

"All parameters"
function allparams()
  φ = Params(scenelen = uniform(10:100),
             width = 100,
             height = 100,
             niterations = uniform([1000, 2000, 5000, 10000, 50000]),
            #  niterations = 10,
             normalizeimgs = bernoulli(0.5, Bool),
             addfloor = bernoulli(0.5, Bool),
             imagesperbatch = uniform(1:10))
  merge(φ, runparams(), optparams(), netparams())
end

"Generate a Lens map"
function genlmap(φ)
  # Tensorboard
  str = linearstring(φ, :runname, :midlen, :niterations, :scenelen, :normalizeimgs, :η, :opt, :imagesperbatch)
  tblogger = TBLogger(joinpath(φ.logdir, str))
  function tblogloss(data)
    with_logger(tblogger) do
      log_value(tblogger, "loss", data.loss; step = data.i)
    end
  end

  # TB log images
  logimg_(data) = log_image(tblogger, "targetimg", data.neural_img, HW; step = data.i)
  logdeepimg_(data) = log_image(tblogger, "deepimg", data.img, HW; step = data.i)
  logimg = everyn(logimg_, 4 * φ.imagesperbatch)
  logdeepimg = everyn(logdeepimg_, 4 * φ.imagesperbatch)
  

  # save network opt scene
  savenet_ = incsave(joinpath(φ.logdir, "net.bson"); verbose = true,
                    save = BSON.bson) ∘ (x -> Dict(:net => x.net, :deepscene => x.deepscene))
  savenet = everyn(savenet_, 50)
  
  # UnicodePlot Nueral Scene
  plotscene = everyn(unicodeplotmat ∘ (x -> x.neural_img), 4 * φ.imagesperbatch)

  # Stopping
  stop = stopnanorinf ∘ (x -> x.loss)

  # Stop when converged
  stopconv = everyn(stopconverged(; verbose = true) ∘ (x -> (x.i, x.loss)), 50)

  # Show Progress
  sp = showprogress(φ.niterations)

  # Plos los
  sl = Callbacks.plotscalar(; width = 100, height = 30) ∘ (x -> (x = x.i, y = x.loss))

  # The lensmap
  lmap = (BatchLoop => runall(stop, plotscene, logimg, logdeepimg),
          TrainLoop => runall(sp, sl, savenet, stopconv, tblogloss))
end

"Generate the network from parameter values"
function gennet(φ, deepscene)
  inlen = linearlength(Vector{Float64}, (Ray{Point3, Point3}, deepscene))
  midlen = φ.midlen
  outlen = 1
  hiddenlayers = [Dense(midlen, midlen, φ.activation) for i = 1:φ.nhidden]
  trackednet = Flux.Chain(Flux.Dense(inlen, midlen, φ.activation),
                          hiddenlayers...,
                          Flux.Dense(midlen, outlen, φ.activation))
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
                      datarv = gendata(; scene = RayTrace.example_spheres(; addfloor = φ.addfloor),
                                         render_params = render_params),
                      normalize = normalize)
end

function testhyper()
  p = rand(allparams())
  mkpath(p.logdir)
  infer(p)
end

end