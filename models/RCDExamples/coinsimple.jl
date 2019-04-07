using Omega 
using Plots

faircoin = bernoulli(0.5, Bool)
headsbiased = bernoulli(0.5, Bool)
function weight_(ω)
  weight = if faircoin(ω)
    0.5
  elseif headsbiased(ω)
    0.6
  else
    0.4
  end
end
weight = ciid(weight_)
coin = bernoulli(weight, Bool)
coinrcd = coin ∥ (faircoin, headsbiased)

# Compute the expectation using nsamples
meannsamples = 100000
probdist = Omega.samplemeanᵣ(coinrcd, meannsamples)

# Draw nsamples from the conditional expectation
nsamples = 100
probsamples = [rand(probdist) for i = 1:nsamples]

fig = histogram(probsamples, nbins=100, xlims=[0.0, 1.0], normalize=true, size=(400, 300), label="")
savefig(fig, joinpath(PLOTSPATH, "coindist.png"))
