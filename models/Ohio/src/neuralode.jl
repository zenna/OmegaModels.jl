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

export main

# ## The Data
const FT = Float32

normalize!(x) = x .= normalize(x)
normalize(x) = length(x) == 1 ? x : (x .- mean(x)) ./ std(x)

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
  tspan = (0.0f0, 15f0)
  t = range(tspan[1], tspan[2], length = datasize)
  n_ode(x) = neural_ode(dudt, x, tspan, Tsit5(), saveat = t, reltol = 1e-7, abstol = 1e-9)

  train(; n_ode = n_ode, dudt = dudt, data = data)
end

end