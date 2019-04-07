using Omega
using UnicodePlots

# Use a Beta distribution for a prior
weight = β(2.0, 2.0)

# Draw 1000 samples from the prior
beta_samples = rand(weight, 10000)

# Visualize the prior of the weight
UnicodePlots.histogram(beta_samples)

# Construct a distribution over coinflips
nflips = 4
coinflips = [bernoulli(weight, Bool) for i = 1:nflips]ᵣ

# `coinflips` is a `RandVar` and hence we can sample from it with `rand` 
rand(coinflips)

# First create some fake data
observations = [true, true, true, false]

# and then use `rand` to draw conditional samples:
weight_samples = rand(weight, coinflips ==ᵣ observations, 10000; alg = RejectionSample)

# Vizualise the conditional (aka, posterior) 
UnicodePlots.histogram(weight_samples)

# Observe that our belief about the weight has now changed.
# We are more convinced the coin is biased towards heads (`true`).