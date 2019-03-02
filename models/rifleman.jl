# Causal Model taken from Causality J. Pearl Page 212
using Omega

# The court orders an executiion with probabilit p
p = 0.5
courtorderexecution = bernoulli(p, Bool)

# With probability Q the rifleman is nervou and fires
q = 0.5
Aisnervous = bernoulli(q, Bool)

# Rifleman A and B both fire if the court orders the exectuion
riflemanA = courtorderexecution | Aisnervous
riflemanB = courtorderexecution

# Prisoner dies if either shoorts
dead = riflemanA | riflemanB

# What's the probability that prisoner is still alive if A had not shot,
# given that he is actually dead?
cdead = replace(dead, riflemanA => false)
mean(rand(cdead, dead, 10000; alg = RejectionSample))