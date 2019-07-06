module BlackScholes

using Omega
using Omega.Prim: samplemeanᵣ
using Plots
using Lens
using Statistics
using Dates

export bsmmc, simrv, diff_σ, sampleprior

# TODO
# Be able to use many observations
# Split plot into prior, one observation, many

# Some visualization functions
const FIGHOME = "figures"
plothist(samples) = histogram(samples)
plotseries(samples) = plot(samples)

# This model simulates the Black-Scholes differential equations

"""

T: time to maturity (in years)
r = riskfree rate of interest
σ: volatility
nsteps: number of time steps

"""
function bsmmc(ω, σ, T = 0.5, nsteps = 16, S = 202.73, r = 0.025)  # initial stock price)
  Δt = T/nsteps
  # series = [S]
  for i = 1:nsteps
    # z = normal(ω, 0, 1)
    z = randn(ω)
    S = S * exp((r - 0.5σ^2) * Δt + (σ * sqrt(Δt) * z))
    # push!(series, S)
  end
  S
  # series
end

# Amazon data
strikedate = Date(2019, 9, 3)
today_ = Date(2019, 7, 3)
const tradingdays = 252             # Number of trading days per year
T = (strikedate - today_).value / 365
nsteps = Int(floor(T * tradingdays))
AAPLS = 204.41
data = [(K = 100.0, c = 104.15, σ = .6895, T = T),
        (K = 120.0, c = 76.03, σ = .5586, T = T),
        (K = 190.0, c = 18.35, σ = .2632, T = T)]


data = [(K = 100.0, c = 104.15, σ = .6895, T = T)]

      
o1 = data[1]  
# We use priors over \sigma, which represents the volatility of the model
const σ = uniform(0.0, 1.0)
const r = 0.02                     # risk-free

# Now we create random variables for the time series, and the value of the stock at time T
const simrv = ciid(bsmmc, σ, o1.T, nsteps, AAPLS, r)
const lastsim =  lift(last)(simrv)

# Let's draw some prior samples from the model
sampleprior() = rand(simrv, 10)
#   nb fig =  plot(samplesprior(); xlabel = "time", ylabel = "S", legend = false)
# savefig(fig, joinpath(FIGHOME, "bsseries1.pdf"))

# Single obseration
const diff = lift(max)(lastsim - o1.K, 0)
const diff_σ =  rid(diff, σ)  
const diffexp = samplemeanᵣ(diff_σ, 1000000)
const C_discount = diffexp * exp(-r*o1.T)

const c1 = C_discount ==ₛ o1.c 
run(; n = 1000, alg = HMCFAST, kwargs...) =
  @leval HMCFASTLoop => default_cbs(n) rand(σ, C_discount ==ₛ o1.c, n; alg = alg, kwargs...)

runsilent(; n = 1000, alg = SSMH, kwargs...) =
  rand(σ, diffexp ==ₛ o1.c, n; alg = alg, kwargs...)  


"Histogram of prior vs posterior samples over σ"
function volatilityplot()
  postsamples = run(; n = 1000)
  priorsamples = rand(σ, 1000)
  histogram([priorsamples, postsamples], fillalpha = 0.5, normalize = true, label = ["Prior", "Posterior"])
end
# savefig(fig, joinpath(FIGHOME, "bshist1.pdf"))

# Multiple observations
function diffmulti_(ω, ks)``
  ls = lastsim(ω)
  [max(ls - k, 0) for k in ks]
end

disp(x; msg = "") = (println(msg, "value : ", x); x)

const nobs = length(data)
const ks = [o.K for o in data] 
const cs = [o.c for o in data]
const diffmulti = ciid(diffmulti_, ks)
const diffmulti_σ = rid(diffmulti, σ)
const diffmultiexp_ =  samplemeanᵣ(diffmulti_σ, 100000000)
const diffmultiexp = lift(disp)(diffmultiexp_)
const diffmultiexpnoise = diffmultiexp + normal(0, 0.01, (nobs,))


const condition = (samplemeanᵣ(diffmulti_σ, 10000) + normal(0, 0.01, (nobs,)))  ==ₛ cs
runmulti(; n = 1000, alg = SSMH, kwargs...) =
  @leval SSMHLoop => default_cbs(n) rand(σ, diffmultiexpnoise ==ₛ cs, n; alg = alg, kwargs...)

runmultisilent(; n = 1000, alg = Replica, kwargs...) =
  rand(σ, diffmultiexpnoise ==ₛ cs, n; alg = alg, kwargs...)

# runmulti() = @leval SSMHLoop => default_cbs(1000) rand(σ, diffmultiexpnoise ==ₛ cs, 1000; alg = Replica)

end # module