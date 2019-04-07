using Omega
using Plots: lineplot
using Distributions

"Plot the pdf of `beta` distribution with parameters `α` and `β`"
plotbeta(α, β) = Plots.plot(i->Distributions.pdf(Beta(α, β), i), 0.0001, 0.999)

# Beta distribution with independent uniforms for  α and β parameters
α = uniform(0.001, 5.0)
β = uniform(0.001, 5.0)
b = betarv(α, β)

# Random conditional distribution given both parameters
brid = rid(b, (α, β)ᵣ)

m = lmean(brid)

# Sample α and β after conditioning the expectation of brcd
@leval Omega.Inference.SSMHLoop => Omega.Inference.plotloss() samples = rand((α, β, m), m ==ₛ 0.75, 1000, alg = SSMH)

plotbeta(rand(samples)[1:2]...)
