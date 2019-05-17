module NeuralODE
using Flux
using Flux.Tracker: gradient, update!
import Flux.Tracker
using DiffEqFlux
using DifferentialEquations
using Lens
using Callbacks
using Statistics
using ..ParseData: expatient, glucoselevels, carblevels
using ..Train: train!, TrainLoop
using Plots
unicodeplots()
# using UnicodePlots: lineplot, lineplot!

# Data
const FT = Float32

# TODO
# How to subsample data (should i?)
# Convert datetimes to timesteps
# Count for ueven spacing in time?

# x init at data init
# x Callback plot prediction

# Callback save weights

# Make probabilistic
# What are the queries


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

# A neural ODE is an ODE where a neural network defines the derivative function
dudt = Chain(Dense(ndim, 50, tanh), Dense(50, ndim))
tspan = (0.0f0, 1.5f0)
t = range(tspan[1], tspan[2], length = datasize)
n_ode(x) = neural_ode(dudt, x, tspan, Tsit5(), saveat = t, reltol = 1e-7, abstol = 1e-9)

# Use the L2 loss of the network's output against the time series data:
# u0 = Float32[2.; 0.]
# Training

function train(; opt = ADAM(0.001))
  datait = Iterators.repeated((), 1000)
  # cb = function () # callback function to observe training
  #   display(loss_n_ode())
  #   # plot current prediction against data
  #   cur_pred = Flux.data(predict_n_ode())
  #   pl = scatter(0.0:0.1:10.0,ode_data[1,:],label="data")
  #   scatter!(pl,0.0:0.1:10.0,cur_pred[1,:],label="prediction")
  #   plot(pl)
  # end
  # # Display the ODE with the initial parameter values.
  # cb()
  @show data
  # @show u0 = normalize(rand(Float32, ndim))
  u0 = data[:, 1] # Initialize where data starts
  predict_n_ode() = n_ode(u0)
  loss_n_ode() = sum(abs2, data .- predict_n_ode())
  ps = Flux.params(dudt)

  # Callbacks

  # Show predictions
  function plotpred(data_)
    preddata = predict_n_ode()
    # @show size(data)
    # @show size(preddata)
    display(plot([data; Tracker.data(preddata)], size = (2000, 300)))
  end
  # Show progress
  sp = showprogress(length(datait))

  # Show loss
  pl = plotscalar() ∘ (x -> (x = x.i, y = x.loss.data))

  # Do training
  @leval TrainLoop => runall([sp, pl, everyn(plotpred, 10)]) train!(loss_n_ode, ps, datait, opt)
end

end