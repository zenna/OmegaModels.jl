using Omega
using Divergences
using StatsBase: normalize, histogram

"Compute the histogram of a distribution"
histogram(x::RandVar, n = 10000) = histogram(rand(x, n))
histogram(samples::Vector; bounds = 0:0.1:1) =
  normalize(fit(Histogram, samples, bounds)).weights
const kldiv = KullbackLeibler()
kl(x, y) = evaluate(kldiv, x, y)
kl(x::RandVar, y::RandVar) = kl(histogram(x), histogram(y))

# Lifted KL
lkl(x, y) = ciid(w -> kl(x(w), y))

# x is the distribution to approximate
x = normal(0.0, 1.0) 

# y is the approximating family
a = uniform(0.001, 5.0)
b = uniform(0.001, 5.0)
y = betarv(a, b)

ridyab = rid(x, a, b) 

# Divergence between x and y 
δxy = lkl(ridyab, x)

# Threshold
δ = 0.01

# Draw conditional samples
ab_ = rand((a, b), δxy < δ)

ys = rand(y, δxy < δ, 1000)


# How to show the examples
