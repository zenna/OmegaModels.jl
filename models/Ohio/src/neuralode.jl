module NeuralODE
using Flux
using Flux.Tracker: gradient, update!
import Flux.Tracker
using DiffEqFlux
using DifferentialEquations
using Lens
using Statistics
using ..ParseData: expatient, glucoselevels, carblevels
using ..Train: train
using RunTools
import Statistics
using Omega: uniform, normal, ciid

export main, infer, testhyper

# ## The Data

normalize!(x) = x .= normalize(x)
normalize(x) = length(x) == 1 ? x : (x .- Statistics.mean(x)) ./ Statistics.std(x)

function getdata(; FT = Float32, datasize = 100, ndim = 2)
  # Extract a sequence of float data from raw values
  patientdata = expatient()
  bigglucosedata = FT.(glucoselevels(patientdata))
  glucosedata = [bigglucosedata[i] for i in Int.(floor.(range(1, length(bigglucosedata), length = datasize)))]
  normalize!(glucosedata)

  # glucosedata = rand(ndim, datasize)

  bigcarbdata = FT.(carblevels(patientdata))  
  carbdata = [bigcarbdata[i] for i in Int.(floor.(range(1, length(bigcarbdata), length = datasize)))]
  normalize!(carbdata)

  data = hcat(glucosedata, carbdata)'
end
# normalize!(data)

# ## The model


"Train the model"
function main()
  # A neural ODE is an ODE where a neural network defines the derivative function
  nhidden = 25
  dudt = Chain(Dense(ndim, nhidden, elu), Dense(nhidden, ndim))
  tspan = (0.0f0, 150f0)
  t = range(tspan[1], tspan[2], length = datasize)
  n_ode(x) = neural_ode(dudt, x, tspan, Tsit5(), saveat = t, reltol = 1e-7, abstol = 1e-9)

  train(; n_ode = n_ode, dudt = dudt, data = data, accum = sum, niterations = 1000)
end

## Hyper Parameter Search
function runparams()
  φ = Params()
  φ.train = true
  φ.loadnet = false
  φ.simname = "infer"
  φ.name = "ohio"
  φ.runname = ciid(randrunname)
  φ.tags = ["ohio", "test"]
  φ.logdir = ciid(ω -> logdir(runname = φ.runname(ω), tags = φ.tags))
  φ.runfile = joinpath(dirname(@__FILE__), "..", "scripts", "runscript.jl")
  φ.gitinfo = current_commit(@__FILE__)
  φ
end

"Optimization Parameters"
function optparams()
  Params((η = uniform([0.01, 0.001, 0.0001, 0.0005]),
          opt = uniform([Descent, ADAM])))
end

function netparams()
  Params(midlen = uniform(40:60),
         nhidden = uniform(0:3),
         activation = uniform([relu, elu, selu]))
end


"All parameters"
function allparams()
  φ = Params(niterations = uniform([1000, 2000, 5000, 10000, 50000]),
             tend = uniform([150, 1500]),
             accum = uniform([sum]),
             FT = uniform([Float64]))
  merge(φ, runparams(), optparams(), netparams())
end

"Train the model"
function infer(φ)
  display(φ)
  ndim = 2
  datasize = 100
  data = getdata(; ndim = ndim, FT = φ.FT, datasize = datasize)
  # A neural ODE is an ODE where a neural network defines the derivative function
  dudt = Chain(Dense(ndim, φ.midlen, φ.activation), Dense(φ.midlen, ndim))
  tspan = (φ.FT(0.0), φ.FT(φ.tend))
  t = range(tspan[1], tspan[2], length = datasize)
  n_ode(x) = neural_ode(dudt, x, tspan, Tsit5(), saveat = t, reltol = 1e-7,  abstol = 1e-9)
  train(; n_ode = n_ode, dudt = dudt, data = data, opt = φ.opt(φ.η),
          accum = φ.accum, niterations = φ.niterations, logdir = φ.logdir)
end

function testhyper()
  p = rand(allparams())
  mkpath(p.logdir)
  infer(p)
end


end