"Selection of models showcasing Random Conditional/Interventional Distribution"
module RCDModels
include("beta.jl")       # Condition expectation of beta distribution
include("coinsimple.jl") # Capture uncertainty over probability coin is heads
include("faircoin.jl")   # 
include("simplify.jl")   # 
include("variation.jl")  # Use RCD to check law of total variance
end