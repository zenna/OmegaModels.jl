module Model
using Omega
using Statistics

export landscape, water, observe

nhills = uniform(1:5)

dist(a, b) = sqrt(sum((a .- b).^2))

# 1. Define a generative model over world
function landscape(ω; width = 100, height = 100)
  # grid = zeros(height, width)
  pos = [(i, j) for i = 1:width, j = 1:height]
  peaks = [(uniform(ω, 1, width), uniform(ω, 1, height)) for i = 1:nhills(ω)]
  displ(x) = map(q -> dist(q, x), peaks)
  map(mean ∘ displ, pos)
end

"Generates water"
function water()

end

"Returns an observation of the `world` and position `pos`"
function observe(world, pos)

end

# Create some observation model

# Create the reward model

# Do the counterfactual inference


end
