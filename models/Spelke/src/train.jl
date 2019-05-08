
## Run
# datapath = joinpath(datadir(), "TwoBalls", "TwoBalls_DetectedObjects.csv")
# datapath = joinpath(datadir(), "Balls_3_Clean_Diverge", "Balls_3_Clean_Diverge_DetectedObjects.csv")
# datapath = joinpath(datadir(), "TwoBalls", "TwoBalls_DetectedObjects.csv")
# datapath = joinpath(datadir(), "data", "Balls_2_DivergenceA", "Balls_2_DivergenceA_DetectedObjects.csv")


function train(data::DataFrame, n = 1000, alg = SSMH, kwargs...)
  realvideo = genrealvideo(data)
  video, latentvideo = priors(realvideo)
  samples = rand(video, video ==ₛ realvideo, n; alg = alg, kwargs...)
  # samples = rand(latentvideo, video ==ₛ realvideo, 1000; alg = alg, kwargs...)
  # evalposterior(samples, realvideo, false, true)
  samples
end

exampledata() =
  CSV.read(joinpath(datadir(), "Balls_3_Clean_Diverge", "Balls_3_Clean_Diverge_DetectedObjects.csv"))

train(datapath::String = joinpath(datadir(), "Balls_3_Clean_Diverge", "Balls_3_Clean_Diverge_DetectedObjects.csv")) =
  train(CSV.read(datapath))

