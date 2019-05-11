using Omega
import Omega.Inference: swapsinglesite, normalkernel
using Plots
using Distributions
import UnicodePlots
using ZenUtils

"Sample from truncated distribution"
function condequal(x, y, k; kwargs...)
  proposal = proposal = (rng, ω) -> swapsinglesite(rng, ω) do x 
    normalkernel(rng, x, 1)
  end
  withkernel(k) do
    rand((x, y), x ==ₛ y, 100000; alg=SSMH, proposal=proposal, kwargs...)
  end
end

function sample(αs)
  x = normal(0.0, 1.0)
  y = normal(0.0, 1.0)
  samples = map(αs) do α
    kernel = Omega.kseα(α)
    condequal(x, y, kernel)
  end
end

function subplot(samples, α, plt = Plots.plot())
  x_, y_ = ntranspose(samples)
  marginalhist(x_, y_)
end

function plotequal(samples, αs)
  # l = @layout [a; b; c; d]
  subplots = subplot.(samples, αs)
  plt = plot(subplots..., layout = (1, length(αs)),
      #  title = "Equality in Distribution",
             fmt = :pdf,
             size = (1000, 200),
             title_location=:left)
  savefig(plt, joinpath(ENV["DATADIR"], "Omega", "figures", "equal.pdf"))
  plt
end

function main(; αs = [0.1, 1.0, 10.0, 100.0, 1000.0], samples = sample(αs))
  plotequal(samples, αs)
end

