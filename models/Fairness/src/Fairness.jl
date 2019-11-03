module Fairness


"""
`Hole(original_value [, (lower_bound, upper_bound)])`
"""
function Hole(x, bounds)

end


"""
Step is a categorical distribution
"""
function step(bars)
end

"I'm not sure what this does"
function event(arg...)
end

# Benchmarks
include("bench/therm_u10_b2.jl")

end
