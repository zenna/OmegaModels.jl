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

# Grid world action 
const SHORTNAME = Dict{Symbol, String}(:up => "u", :down => "d", :left => "l", :right => "r")

# Use Levenshtein (String Distance) to compare strings
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
  # Random cells on grid have random reward
  rewards = Dict(GWPos(r[Int(ceil(rand(ω) * 10))], r[Int(ceil(rand(ω) * 10))]) => uniform(ω, 0.0, 10) for i = 1:3)
  SimpleGridWorld(rewards = rewards)
end

gridworld = ciid(gridworld_)

"Distribution over action sequences"
function actionseq_(ω; seed = 12345)
  # Create a grid world
  mdp = gridworld(ω)
  # MCTS is a ranomdized algorithm, but we don't want to do inference over random
  # sampling within it.  To make actionseq a pure function of ω
  # use different rng but fix seed
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

"""Sample from posterior distribution over worlds conditioned on agent acting in
world produding observed action sequence"""
function invplan(; n = 100, alg = Replica, kwargs...)
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
  rand((gridworld, actionseq), actionseq ==ₛ obs, n, alg = alg; kwargs...)
end

end
