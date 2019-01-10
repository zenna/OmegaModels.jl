using Omega
# using Plots
using Distributions
using Plots
import StatPlots


"Sample from truncatedistd distribution"
function truncatedist(x, lb, ub, k, n; kwargs...)
  withkernel(k) do
    rand(x, (lb < x) & (x < ub), n; kwargs...)
  end
end

function sample()
  x = normal(0.0, 1.0)
  αs = [0.1, 1.0, 10.0, 100.0]
  kernels = Omega.kseα.(αs)
  samples = [truncatedist(x, 0.0, 1.0, k, 100000) for k in kernels]
end

function subplot(samples, α, plt = plot())
  density!(samples, label = "a = $α",
                    #  m=(0.001,:auto),
                     style = :auto,
                     w = 2.0)
end

function plotdist(samples, save = false)
  plt = plot(title = "Truncated Normal through Conditioning")
  αs = [0.1, 1.0, 10.0, 100.0]
  foreach((α, s) ->  subplot(s, α, plt), αs, samples)
  save && savefig(plt, joinpath(ENV["DATADIR"], "mu", "figures", "truncatedistd.pdf"))
  plt
end

function main()
  plotdist(sample())
end

main()

