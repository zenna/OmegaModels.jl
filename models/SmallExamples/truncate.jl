using Omega
# using Plots
using Distributions
using Measures
using Printf
using Plots
import StatsPlots


"Sample from truncatedistd distribution"
function truncatedist(x, lb, ub, k, n; kwargs...)
  withkernel(k) do
    rand(x, (lb <ₛ x) & (x <ₛ ub), n; kwargs...)
  end
end

function sample(αs; kwargs...)
  x = normal(0.0, 1.0)
  kernels = Omega.kseα.(αs)
  samples = [truncatedist(x, 0.0, 1.0, k, 100000; kwargs...) for k in kernels]
end

function subplot(samples, α, c, plt = plot())
  αname = @sprintf("\\alpha = 10^%d", trunc(Int, -log10(α)))
  StatsPlots.density!(samples, label = αname, # "\\alpha = $(αname)",
                    #  m=(0.001,:auto),
                     c = c,
                     legend=:topleft,
                    #  top_margin=6mm,
                     style = :solid,
                    #  alpha=0.75,
                    #  fill=(0,),
                     w = 9,
                     titlefontsize=24,
                     legendfontsize=22,
                     tickfontsize=20)
end

function subplot_fill(samples, α, c, plt = plot())
  αname = @sprintf("%.2f", 1/α)
  StatsPlots.density!(samples, label = "\\alpha = $(αname)",
                     c = c,
                     legend=:topleft,
                     style = :solid,
                     alpha=0.75,
                     fill=(0,),
                     w = .1,
                     titlefontsize=27,
                     legendfontsize=24,
                     tickfontsize=18)
end

function plotdist(αs, samples, colors, save = false)
  # plt = plot(title = "Truncated Normal through\n Conditioning");
  plt = plot()
  # alternative color map
  # colors = ["#ff7f00", "#984ea3", "#4daf4a", "#377eb8"]
  foreach((α, s, c) ->  subplot(s, α, c, plt), αs, samples, colors)
  yticks!(plt, [0.0, 0.5, 1.0])
  save
  savefig(plt, joinpath(ENV["DATADIR"], "mu", "figures", "truncatedistd.pdf"))
  plt
end

function createcolors(αs)
  PlotUtils.clibrary(:cmocean)
  C(g::ColorGradient) = RGB[g[z] for z=range(0,stop=1,length=length(αs)+1)]
  (cgrad(:amp) |> C)[2:end]
end

function main(; kwargs...)
  
  αs = [0.1, 1.0, 10.0, 100.0]
  colors = createcolors(αs)
  plotdist(αs, sample(αs; kwargs...), colors)
end

main(; alg = NUTS)

