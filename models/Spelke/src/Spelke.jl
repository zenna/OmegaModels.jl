module Spelke
using Omega
import Omega
using UnicodePlots
using CSV
using DataFrames
# using RunTools
using ArgParse
using PDMats
using Statistics: std

export train, viz, draw, exampledata

datadir() = joinpath(@__DIR__, "..", "data")

include("distances.jl")     # Distance measures
include("model.jl")         # The model
include("draw.jl")          # Drawing / Vizualisation
include("train.jl")         # Training
include("evals.jl")         # Evaluate the posterior


end