
# Simulation of co2 vs time with volatility σ
const σ = uniform(0.0, 1.0)
const μ = normal(0, 1)
const sim_end = 2035
const co2_sim = ~ ω -> weiner(ω, μ(ω), σ(ω), co2_data[1959], length(1959:sim_end))
const co2_sim_historical = co2_sim[1:length(co2_data)]

# Model Parameters
const Θ = randtuple((σ, μ))

# Posterior over parameters given simulation = data
const datacond = co2_sim_historical ==ₛ asvec(co2_data, 1959:2018)
const Θ_cond = cond(Θ, datacond)

# Posterior over parameters given there will be a 25 percent increase 
const idx_2018 = 2018 - 1959 + 1
const increase = co2_sim[end] / co2_sim[idx_2018]
const Θ_cond_2 = cond(Θ, datacond & increase ==ₛ 1.25)

# Expected increase will be 25 percent
const Θ_cond_rcd = cond(Θ, datacond & meanᵣ(rid(increase, Θ)) ==ₛ 1.25)