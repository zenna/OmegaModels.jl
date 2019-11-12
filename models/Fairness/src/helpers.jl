
"""
`Hole(original_value [, (lower_bound, upper_bound)])`
"""
function Hole(ω, x, bounds)
  @assert false
end


"Hole without constraints"
Hole(ω, x; σ = 1) = normal(ω, x, σ)

"Density is a step function (piecewise constant)"
function steps(ω, bars)
  total = 0.0
  r = uniform(ω, 0, 1)
  for bar in bars
    total += bar[3]
    if r <= total
      return uniform(ω, bar[1], bar[2])
    end
  end
  @assert false "This statement is reached with 0 probability"
end

# def step(bars):
#     # bars is a list of (min,max,pmass)
#     sanity = sum([bar[2] for bar in bars])
#     assert 1. - sanity < 0.00001 , "bars sum to " + str(sanity)
#     r = random()
#     total = Fraction(0)
#     for bar in bars:
#         total += Fraction(bar[2])
#         if r <= total:
#             return random() * (bar[1] - bar[0]) + bar[0]
#     assert False, "This statement is reached with 0 probability

gaussian(ω, μ, σ) = normal(ω, μ, σ)

"I'm not sure what this does"
function event(arg...)
end
