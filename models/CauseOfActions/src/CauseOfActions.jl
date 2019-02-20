"Infer the cause of ones actions"
module CauseOfActions
using Omega
using Images
using ColorSchemes
using POMDPs
using POMDPModelTools
using POMDPSimulators
using StaticArrays
using Parameters
using Random
using MCTS
using StringDistances

const MAPPATH = joinpath(dirname(pathof(CauseOfActions)), "..", "maps")

include("gridworld.jl")

# So the question here is what to do about the fact that
# MCTS uses lots fo randomness 

# 1. do inference over all actions
# not sure it leads to right result:
# e.g. 

# x = normal(0, 1)
# y = normal(x, 1)
# ymean = mean(rcd(y, x))

# souppose ymean_ = rand(ymean) = normal(0.4, 1)
# suppose we want to condiiton this to have expectation 2,
# we can control the samples such that this is true,
# but that would be wrong.

# Alternatively, 
# 2. dont do inference oer these actions and ignore randoness in output
# 3. Fix random seed so given ω its determinisitc but only do inference over orld

const SHORTNAME = Dict{Symbol, String}(:up => "u", :down => "d", :left => "l", :right => "r")

function Omega.d(x::Array{<:Symbol}, y::Array{<:Symbol}; stringdist = StringDistances.Levenshtein())
  x_ = join(map(s -> SHORTNAME[s], x))
  y_ = join(map(s -> SHORTNAME[s], y))
  res = 1 - compare(stringdist, x_, y_)
  x_, y_, res
  res
end

"Distribution over grid worlds"
function gridworld_(ω; seed = 12345)
  r = collect(1:10)
  # a = Int(ceil(rand(ω) * 10))
  # b = Int(ceil(rand(ω) * 10))
  # c = Int(ceil(rand(ω) * 10))
  # uniform(ω, r)
  rewards = Dict(GWPos(r[Int(ceil(rand(ω) * 10))], r[Int(ceil(rand(ω) * 10))]) => uniform(ω, 0.0, 10) for i = 1:3)
  SimpleGridWorld(rewards = rewards)
end

gridworld = ciid(gridworld_)

"Distribution over grid worlds"
function actionseq_(ω; seed = 12345)
  # Create a grid world
  mdp = gridworld(ω)
  rng  = Random.MersenneTwister(seed)
  solver = MCTSSolver(n_iterations=50, depth=20, exploration_constant=5.0, rng = rng)
  policy = solve(solver, mdp)
  actions = Symbol[]
  for (s, a, r) in stepthrough(mdp, policy, "s,a,r", max_steps=10; rng = rng)
    push!(actions, a)
  end
  actions
end

const actionseq = ciid(actionseq_)

function invplan(; n = 100, kwargs...)
  obs = [:up,
        :up,   
        :left, 
        :up,   
        :up,   
        :down, 
        :right,
        :up,   
        :left, 
        :up]  
  rand((gridworld, actionseq), actionseq ==ₛ obs, n, alg = Replica; kwargs...)
end

# const gridworld = ciid(gridworld_)
# const policy = lift(solve)(gridworld)


# "Convert an image to a map"
# function tomap(img, cscheme::ColorScheme = ColorSchemes.RdBu_9)
#   map(img) do c
#     getinverse(cscheme, c)
#   end
# end

# function test()
#   world1 = load(joinpath(MAPPATH, "world1.png"))
# end


# rand(gridworld, actons )

# What happens if:
# U use a distribution over worlds (with omega),
# And have a MDP
# You construct a distribution over MDPS

# What if I have a distribution over worlds\
# And a distribution over the uncertainty, e.g. the transition probabilities

# What about the 
end
