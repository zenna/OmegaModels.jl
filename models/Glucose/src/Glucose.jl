module Glucose
using Omega
using Flux
using DataFrames
using CSV
using Plots
using LinearAlgebra: normalize

# TODO: Show a plot of prediction error
# TODO: 
# Feel a bit uneasy about htis, why? 
# data is very small
# Shouldn't we be sharing weights
# Data isn't equally spaced
# Just not a very good model

# Path to Glucose data
GLUCOSEDIR = joinpath(dirname(pathof(Glucose)), "..")
DATADIR = joinpath(GLUCOSEDIR, "data")
FIGURESDIR = joinpath(GLUCOSEDIR, "figures")

include("plots.jl")

dist(x, y) = (x - y)^2.0

# The model

"Simulates `nsteps` of RNN, returns output at each step"
function rnn_(ω, f, nsteps, h1_size) 
  # h = zeros(h1_size) # What should this be?
  xs = []
  input = zeros(h1_size + 1)
  for i = 1:nsteps
    input = f(input)
    x = input[1] # Takes single element of hidden layer
    push!(xs, x)
  end
  [xs...]
end

"RNN model for patient `personid`"
function model(personid; nsteps = 20, h1_size = 10, h2_size = 30)
  function F_(ω)
    other = Chain(
              Flux.Dense(ω[@id][personid][2], h1_size, h2_size, Flux.relu),
              Flux.Dense(ω[@id][personid][3], h2_size, 1, Flux.sigmoid))
    Chain(
      Flux.Dense(ω[@id][personid][1], 1 + h1_size, h1_size, Flux.relu),
      h -> vcat(other(h), h))
  end
  ciid(F_)
end

"RandVar over nsteps of RNN simulation"
sim(f, nsteps, h1_size) = ciid(ω -> rnn_(ω, f(ω), nsteps, h1_size))

#   # Create one network per person
#   fs = [ciid(ω -> F_(ω,  i)) for i = 1:npatients]

#   # Create one simulation RandVar for each patient
#   sims = [ciid(ω -> rnn_(ω, f(ω), nsteps, h1_size)) for f in fs]
#   # # Take average over time
#   # meansims = ω -> mean.(sims(ω))
#   # sims, meansims
# end

"Averages over time"
meansims(sims) = ciid(ω -> mean.(sims(ω)))

"Data filtered to patient `personid`, on measurement `measure` sorted in time"
function filterdata(data, personid, measure = 807)
  people = groupby(data, :Id)
  p1 = people[personid]
  p2 = filter(row -> row[:Measure] == measure, p1)
  sort(p2, :Time,)
end

"Retruns normalized vector of data Vector{T}"
function timeseries(persondata, nsteps, ::Type{T} = Float64) where T 
  range = 1:min(nsteps, nrow(persondata))
  normalize(T.(persondata[:Value]))[range]
end

"Sequence of observed data for patient `personid`"
observations(personid, nsteps; data = loaddata()) = timeseries(filterdata(data, personid), nsteps)

datalen(personid; data = loaddata(), kwargs...) = size(filterdata(data, personid; kwargs...), 1)


"Conditioning RandVar: sim == personid.data"
function datacond(sim, personid, nsteps; data = loaddata())
  # Get data for person i
  personiddata = filterdata(data, personid)
  obvglucose = timeseries(personiddata, nsteps)

  # Return Condition
  datacond = sim[range] == obvglucose
  datacond, obvglucose
end

"Load the data"
loaddata() = CSV.read(joinpath(DATADIR, "glucosedata.csv"))

"RNN model conditioned on data from `personid`"
function conditioned_model(personid; data = loaddata(), nsteps, h1_size, modelkwargs...)
  f = model(personid; nsteps = nsteps, h1_size = h1_size, modelkwargs...)
  sim_ = sim(f, nsteps, h1_size)
  obs = observations(personid, nsteps; data = data)
  sim ==ₛ obs
end

"Sample posterier without ties between patients"
function sample_posterior(personid, nsteps = 20; n = 5000, alg = Replica, algargs...)
  m = Glucose.conditioned_model(personid; nsteps = nsteps)
  rand(m, n; alg = alg, algargs...)
end

"Sample posterier with ties between patients"
function sample_tied_posterior()
  data = loaddata()
  sims, simsω, (obvglucose_3, obvglucose_4) = Omega.withkernel(Omega.kseα(200)) do
    h1, h2 = 25, 25
    npatients = 5
    nsteps = 20
    sims, meansims = model(nsteps, h1, h2)
    nsteps = 20
    n = 1000
    y_3, obvglucose_3 = datacond(data, sims[3], 3, nsteps)
    y_4, obvglucose_4 = datacond(data, sims[4], 4, 1)
    δ = 0.001
    ties = [lift(dist)(meansims[i], meansims[j]) < δ for i = 3:3, j = 1:npatients if i != j]
    simsω = rand(SimpleΩ{Vector{Int}, Flux.TrackedArray}, (y_4 & y_3) & ((&)(ties...)), HMCFAST,
                  n=n, stepsize = 0.01);
    sims, simsω, (obvglucose_3, obvglucose_4)
  end
end

# Plots

"Compare Data, rnn prior, rnn posterior, and predictive posterior"
function predictionplot(personid; n = 10, alg = Replica, h1_size = 10, modelkwargs = (), algkwargs = ())
  nsteps = datalen(personid)

  f = model(personid; nsteps = nsteps, modelkwargs...)
  priorsim = sim(f, nsteps, h1_size)
  predsim = sim(f, nsteps * 2, h1_size)

  # Simulate from Prior
  priorsim_ = rand(priorsim)

  obs = observations(personid, nsteps)

  # Posterior simulations
  postsim_, predsim_ = withkernel(kseα(10000)) do
    rand((priorsim, predsim), priorsim ==ₛ obs, n; alg = alg, algkwargs...)[end]
  end

  plot([obs, priorsim_, postsim_, predsim_], label = ["Observation  ", "Prior", "Posterior", "Prediction"])
end

predictionplot(5, n = 5000)


end
