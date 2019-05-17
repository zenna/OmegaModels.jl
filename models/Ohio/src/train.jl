module Train
using Flux
using Flux.Tracker: gradient, update!
using Lens

export train!, TrainLoop

struct TrainLoop end

"Like `Flux.train!` but using Lens instead of callbacks"
function train!(loss, ps, data, opt)
  ps = Flux.Params(ps)
  for (i, d) in enumerate(data)
    try
      gs = Flux.gradient(ps) do
        @show l = loss(d...)
        lens(TrainLoop, (loss = l, i = i))
        l
      end
      update!(opt, ps, gs)
    catch ex
      if ex isa Flux.Optimise.StopException
        break
      else
        rethrow(ex)
      end
    end
  end
end

end