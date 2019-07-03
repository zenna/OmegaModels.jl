module BlackScholes

using Omega
using Omega.Prim: samplemeanᵣ
using Plots
using Lens
using Statistics
using Dates

export bsmmc, simrv, diff_σ

# TODO
# Do inference with real put data 
# Be able to use many observations
# Split plot into prior, one observation, many

# Some visualization functions
const FIGHOME = "/home/zenna/repos/papers/The-Random-Conditional-Distribution-Paper/figures/"
plothist(samples) = histogram(samples)
plotseries(samples) = plot(samples)

# This model simulates the Black-Scholes differential equations

# # Brownion motion
# `simulate` below performs brownian motion to simulate a time series

# ```math
# t_{n+1} + normal(t_n, \sigma)`
# ```

# """Black Scholes Model (Monte Carlo)
# """
# function bsmmc(ω, σ; nsteps = 16, Δt = 1, S = 202.73)  # initial stock price)
#   Ss = Float64[S]
#   for i = 1:nsteps
#     S += normal(ω, 0, sqrt(1/nsteps) * σ)
#     # ΔS = S * (μ * Δt + σ * s 
#     # push!(Ss, S)
#   end
#   Ss
# end

"""

T: time to maturity (in years)
r = riskfree rate of interest
σ: volatility
nsteps: number of time steps

"""
function bsmmc(ω, σ, T = 0.5, nsteps = 16, S = 202.73, r = 0.025)  # initial stock price)
  Δt = T/nsteps
  series = Float64[S]
  for i = 1:nsteps
    z = normal(ω, 0, 1)
    S = S * exp((r - 0.5σ^2) * Δt + (σ * sqrt(Δt) * z))
    push!(series, S)
  end
  series
end

# We use priors over \sigma, which represents the volatility of the model
# const σ = uniform(0.0, 5.0)

# Fakebook Stock details
# const K = 107.0474967956543       # Striking price
# const volatility = 0.2573358803663978   # Returns
# const σ = constant(volatility)
# const T = 0.5                       # Time (in years)
# const r = 0.025                     # risk-free
# const tradingdays = 252             # Number of trading days per year
# const nsteps = Int(T * tradingdays)
# const S = 101.94999694824219        # Initial stock price

# Amazon data
strikedate = Date(2019, 9, 3)
today_ = Date(2019, 7, 3)
const tradingdays = 252             # Number of trading days per year
T = (strikedate - today_).value / 365
nsteps = Int(floor(T * tradingdays))
AAPLS = 204.41
data = [(K = 100.0, c = 104.15, σ = .6895, T = T, S = AAPLS),
        (K = 120.0, c = 76.03, σ = .5586, T = T, S = AAPLS),
        (K = 190.0, c = 18.35, σ = .2632, T = T, S = AAPLS)]
        const K = 107.0474967956543       # Striking price

        const volatility = 0.2573358803663978   # Returns

o1 = data[1]
# const σ = constant(o1.σ)
const σ = uniform(0.0, 5.0)
const r = 0.025                     # risk-free
const S = 101.94999694824219        # Initial stock price


# Now we create random variables for the time series, and the value of the stock at time T
const simrv = ciid(bsmmc, σ, o1.T, nsteps, o1.S, r)
const lastsim = lift(last)(simrv)

# Let's draw some prior samples from the model
sampleprior() = rand(simrv, 2)
# nb plot(samplesprior())
# savefig(fig, joinpath(FIGHOME, "bshist1.pdf"))

# Single obseration
const diff = lift(max)(lastsim - o1.K, 0)
const diff_σ =  rid(diff, σ)  
const diffexp = samplemeanᵣ(diff_σ, 1000)
const C_discount = diffexp * exp(-r*o1.T)

run(; n = 1000, alg = SSMH, kwargs...) =
  @leval SSMHLoop => default_cbs(n) rand(σ, diffexp ==ₛ o1.c, n; alg = alg, kwargs...)

"Histogram of prior vs posterior samples over σ"
function volatilityplot()
  postsamples = run(; n = 1000)
  priorsamples = rand(σ, 1000)
  histogram([priorsamples, postsamples], fillalpha = 0.5, normalize = true, label = ["Prior", "Posterior"])
end
# savefig(fig, joinpath(FIGHOME, "bshist1.pdf"))

# Multiple observations
function diffmulti_(ω, ks)
  ls = lastsim(ω)
  [max(ls - k, 0) for k in ks]
end

const nobs = 3
const diffmulti = ciid(diffmulti_, [1.0, 2.0, 3.0])
const diffmulti_σ = rid(diffmulti, σ)
const diffmultiexp =  samplemeanᵣ(diffmulti_σ, 1000)
const diffmultiexpnoise = diffmultiexp + normal(0, 0.01, (nobs,))

# Create fake data
"Generate fake data where `σ` is `σc`"
# function genfakedata(; σc = 3.0)
#   fakedatasamples = rand(diffmultiexp, 500)
#   @show fakedata = mean(fakedatasamples)
# end

runmulti() = @leval SSMHLoop => default_cbs(1000) rand(σ, diffmultiexpnoise ==ₛ [0.75901, 0.450418, 0.247978], 1000; alg = Replica)

end # module