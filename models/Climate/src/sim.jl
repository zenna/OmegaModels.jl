
"""

T: time to maturity (in years)
r = riskfree rate of interest
σ: volatility
nsteps: number of time steps

"""
function brownian(ω, σ, T = 0.5, nsteps = 16, S = 202.73, r = 0.025) 
  series = [S]
  for i = 1:nsteps - 1
    z = randn(ω)
    S = S * exp((r - 0.5σ^2) + (σ * z))
    push!(series, S)
  end
  # S
  series
end

"Weiner process with drift"
function weiner(ω, μ, σ, S, nsteps) 
  series = [S]
  for i = 1:nsteps - 1
    z = normal(ω, μ, σ)
    S = S + z
    push!(series, S)
  end
  # S
  series
end