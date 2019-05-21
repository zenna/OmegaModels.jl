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
using Omega: uniform, normal

export main, infer, testhyper

# ## The Data
const FT = Float32

normalize!(x) = x .= normalize(x)
normalize(x) = length(x) == 1 ? x : (x .- Statistics.mean(x)) ./ Statistics.std(x)

# Extract a sequence of float data from raw values
ndim = 2
patientdata = expatient()
bigglucosedata = FT.(glucoselevels(patientdata))
datasize = 100
glucosedata = [bigglucosedata[i] for i in Int.(floor.(range(1, length(bigglucosedata), length = datasize)))]
normalize!(glucosedata)

# glucosedata = rand(ndim, datasize)

bigcarbdata = FT.(carblevels(patientdata))  
carbdata = [bigcarbdata[i] for i in Int.(floor.(range(1, length(bigcarbdata), length = datasize)))]
normalize!(carbdata)

data = hcat(glucosedata, carbdata)'
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
  Params((η = uniform([0.01, 0.001, 0.0001, 0.00001]),
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
             tend = uniform([1.5f0, 15f0, 150f0, 1500f0]),
             accum = uniform([maximum, Statistics.mean]))
  merge(φ, runparams(), optparams(), netparams())
end

"Train the model"
function infer(φ)
  # A neural ODE is an ODE where a neural network defines the derivative function
  dudt = Chain(Dense(ndim, φ.midlen, φ.activation), Dense(φ.midlen, ndim))
  tspan = (0.0f0, φ.tend)
  t = range(tspan[1], tspan[2], length = datasize)
  n_ode(x) = neural_ode(dudt, x, tspan, Tsit5(), saveat = t, reltol = 1e-7, abstol = 1e-9)
  train(; n_ode = n_ode, dudt = dudt, data = data, opt = φ.opt(φ.η),
          accum = φ.accum, niterations = φ.niterations)
end

testhyper() = infer(rand(allparams()))



end