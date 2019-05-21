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
function genlmap(; data, predict, niterations, dudt, logdir = "")
  is1 = incsave(joinpath(logdir, "gvt.png"); verbose = true, save = (path, fig) -> savefig(fig, path))
  function plotpred(data_)
    preddata = predict()
    plt = plot([data; Tracker.data(preddata)]', size = (2000, 300))
    is1(plt)
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

function train(; n_ode, dudt, data, opt = ADAM(0.0005),
                 niterations, accum, logdir = "")
  datait = Iterators.repeated((), niterations)
  @show data

  u0 = data[:, 1] # Initialize where data starts
  predict_n_ode() = n_ode(u0)
  # loss_n_ode() = sum(abs2, data .- predict_n_ode()) / length(data)
  function loss_n_ode()

    @show size(predict_n_ode())
    @show size(data)
    accum(abs2, data .- predict_n_ode())
  end
  ps = Flux.params(dudt)
  lmap = genlmap(; data = data, predict = predict_n_ode, dudt = dudt,
                   niterations = niterations, logdir = logdir)

  # Do training
  @leval lmap train!(loss_n_ode, ps, datait, opt)
end

end