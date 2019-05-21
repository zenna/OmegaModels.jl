module Train
using Flux
using Flux.Tracker: gradient, update!
using Lens
using Plots
using BSON
using Callbacks
unicodeplots()

include("fluxtrain.jl")
export train!, TrainLoop

"Generate Lens Map"
function genlmap(; data, predict, niterations, dudt)
  function plotpred(data_)
    preddata = predict()
    display(plot([data; Tracker.data(preddata)]', size = (2000, 300)))
  end
  # Show progress
  sp = showprogress(niterations)

  # Show loss
  pl = plotscalar() ∘ (x -> (x = x.i, y = x.loss.data))

  # Save the network
  is = incsave(joinpath("./", "net.bson"); verbose = true)
  savenet_(data) = is(Dict("dudu" => dudt))
  savenet = everyn(savenet_, 10)

  TrainLoop => runall(sp, pl, everyn(plotpred, 10), savenet)
end

function train(; n_ode, dudt, data, opt = ADAM(0.0005), datait = Iterators.repeated((), 10000))
  @show data

  u0 = data[:, 1] # Initialize where data starts
  predict_n_ode() = n_ode(u0)
  # loss_n_ode() = sum(abs2, data .- predict_n_ode()) / length(data)
  loss_n_ode() = maximum(abs2, data .- predict_n_ode())
  ps = Flux.params(dudt)
  lmap = genlmap(; data = data, predict = predict_n_ode, dudt = dudt,
                   niterations = length(datait))

  # Do training
  @leval lmap train!(loss_n_ode, ps, datait, opt)
end

end