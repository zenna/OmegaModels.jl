module Spelke
using Omega
using UnicodePlots
using CSV
using DataFrames
# using RunTools
using ArgParse
using PDMats
using Statistics: std

export train, viz, draw

datadir() = joinpath(@__DIR__, "..", "data")

include("distances.jl")     # Distance measures
include("evals.jl")         # Evaluate the posterior
include("model.jl")         # The model
include("draw.jl")          # Drawing / Vizualisation
include("train.jl")         # Training

"Constructs data-dependent priors"
function priors(realvideo)
  video = ciid(ω -> video_(ω, realvideo, length(realvideo), render))
  latentvideo = ciid(ω -> video_(ω, realvideo, length(realvideo)))
  (video = video, latentvideo = latentvideo) 
end

priors(datapath::String = joinpath(datadir(), "Balls_3_Clean_Diverge", "Balls_3_Clean_Diverge_DetectedObjects.csv")) = priors(genrealvideo(CSV.read(datapath)))

end