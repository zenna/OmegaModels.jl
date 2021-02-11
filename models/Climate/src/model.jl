
# Simulation of co2 vs time with volatility σ
σ = uniform(0.001, 1.0)
co2sim = ~ ω -> brownian(ω, σ(ω), 0.5, length(co2_data), co2_data[1959])

# Simulation of temperature levels
k = uniform(0, 1)     # Unknown period
α = uniform(0, 100)   # Unknown scale factor
tempsim = ~ ω -> map(((t, s),) -> s + α(ω) * sin(t*k(ω)), enumerate(co2sim(ω)))

# Model Parameters
params = randtuple((σ, k, α))

params_cond = cond(params, co2sim ==ₛ asvec(co2_data, 1959:2018))

ΔS =~ ω -> let c = co2sim(ω); c[2035] - c[2016] end

# RCD example
# params_cond ∥ mean(ΔS) in 1..2