"Population Level Inverse Planning and Counterfactuals"
module IslandDispute

using Omega
using Images
using POMDPs
using POMDPModelTools
using POMDPSimulators
using StaticArrays
using Parameters
using Random
using MCTS
using UnicodePlots
using DiscreteValueIteration
import Plots

# Consider a migration dispute between three hypothetical island nations (Figure \ref{fig:islandviz} Left):
# $S$ to the South, $E$ to the East and $N$ to the North.
# The government of $S$ aims to reduce emigration of its population to the $N$, and considers constructing a barrier between $S$ and $N$ (Figure \ref{fig:islandviz} right).

# We model this problem as counterfactional inference in a Markov Decision Process \cite{puterman2014markov} (MDP) model.
# To determine whether the border can be effective:
# = We assume members of the population are rational: that they migrate according to their beliefs about the world and their objectives.
# - We condition on observed migration patterns to infer a posterior belief over the population objectives.
# - In this conditional model, we consider the intervention (adding the barrier) and predict the resulting migration.

# Concretely, we assume the world is a 7 by 6 grid and a world population of $1000$ residents.
#`SimpleGridWorld` defines an MDP which we will solve for each member of the population using POMDPs
include("gridworld.jl")
include("plots.jl")

# Each cell is either land, water, or wall

"Piece of land, water or wall"
struct Terrain
  id::Int
  pos::Vector{GWPos}
end

# Each member of this universe receives cost at every time step
# This cost is subjective and depends on the terrain

"Terrain with cost"
struct CostTerrain
  obj::Terrain
  cost::Float64
end

# Islands: South, North, East
S = Terrain(1, [GWPos(2,1), GWPos(3, 1), GWPos(4, 1), GWPos(3,2)])
N = Terrain(2, [GWPos(3,4), GWPos(2, 5), GWPos(3, 5), GWPos(4,5), GWPos(3,6)])
E = Terrain(3, [GWPos(7,2), GWPos(6, 3), GWPos(7,3), GWPos(7,4)])
islands = (S, N, E)

# wall = Terrain(3, [GWPos(1,3), GWPos(2,3), GWPos(3,3), GWPos(4,3), GWPos(5,3)])

# Wall is another kind of terrain (that is hard to cross)
wall = Terrain(4, [GWPos(1,3), GWPos(2,3), GWPos(3,3), GWPos(4,3), GWPos(5,3),
                   GWPos(1,4), GWPos(1, 5), GWPos(1, 6),
                   GWPos(5,4), GWPos(5, 5), GWPos(5, 6)])

# We'll assume that the cost of being in water or traversing a wall is constant among the population
watercost = -10.0
wallcost = -30

# Construct an inverse map from position to the terrain type
const invmap = Dict()
for ter in [S, N, E, wall]
  for pos in ter.pos
    invmap[pos] = ter
  end
end
  
"Rewards Dict from positiosn to values"
function rewards(objects, defaultcost = watercost)
  rewards_ = Dict{GWPos, Float64}()
  for obj in objects
    for xy in obj.obj.pos
      if xy in keys(rewards_)
        rewards_[xy] += obj.cost
      else
        rewards_[xy] = obj.cost
      end
    end
  end
  rewards_
end

HIGH =  10.0
NEUTRAL = 0.0
LOW = -10.0

# terraincostmat maps where a person is born to their beliefs about all other natiosn
terraincostsmat = [normal(LOW, 3.0) normal(NEUTRAL, 3.0) normal(HIGH, 3.0);
                   normal(LOW, 3.0) normal(HIGH, 3.0) normal(LOW, 3.0);
                   normal(NEUTRAL, 3.0) normal(NEUTRAL, 3.0) normal(NEUTRAL, 3.0)]

terraincostsmatrv = constant(terraincostsmat)
addwall = constant(false)

"Simulate migration for one citizen"
function solveworld(rng, ω; size = (7,6), defrward = watercost,
                            solver = ValueIterationSolver(max_iterations=10, belres=1e-6))
  # Sample the island of birth uniformly along islands
  birthisland = rand(rng, [islands...]) #FIXME? Should this be rng?

  # Within the country sample a birth position
  birthplace_ = rand(rng, birthisland.pos)
  terraincostsmat_ = terraincostsmatrv(ω)

  # Construct a believ about the world: i.e. costs of every terrain for this person
  objects = [CostTerrain(obj, terraincostsmat_[birthisland.id, obj.id](ω)) for obj in islands]

  # Potentially add the wall to their set of beleifs
  if addwall(ω)
    push!(objects, CostTerrain(wall, wallcost))
  end
  # objects = [CostTerrain(obj, terraincosts[(birthisland, obj)](ω)) for obj in islands]

  # Construct a reward function
  rewards_ = rewards(objects)
  for (k, v) in rewards_
    rewards_[k] += randn(rng)
  end

  # Construct an mdp 
  mdp = SimpleGridWorld(size = size,
                        rewards = rewards_,
                        defreward = defrward,
                        initialstate = birthplace_,
                        tprob = 0.99,
                        terminate_from = Set{GWPos}())

  # Solve the mdp to construct a policy
  policy = solve(solver, mdp) # runs value iterations

  # Act according to the policy for 10 time steps
  states = GWPos[]
  rs = Float64[]
  actions = Symbol[]
  # @show objects
  for (s, a, r) in stepthrough(mdp, policy, "s,a,r", max_steps=10; rng = rng)
    push!(actions, a)
    push!(states, s)
    push!(rs, r)
    # @show s
  end
  # @show actions
  # @show states
  # @show rs
  # println("\n\n")
  states
end

"Simulate global migration `npeople` people"
function world(ω, npeople = 100, seed = 12345)
  rng  = Random.MersenneTwister(seed)
  statesseqs = Vector{GWPos}[]
  for i = 1:npeople
    push!(statesseqs, solveworld(rng, ω))
  end
  statesseqs
end

function migrationmovements(allseqs)
  mat = zeros(length(islands), length(islands) + 2)
  for stateseq in allseqs
    birth = invmap[stateseq[1]]
    for state in stateseq
      terr = get(invmap, state, 0)
      if terr == 0
        mat[birth.id, 5] += 1.0
      else
        mat[birth.id, terr.id] += 1.0
      end
    end
  end
  mat
end

wrld = ciid(world)
function runmodel(; usereplmap = false, replmap = Dict())
  samples = usereplmap ? rand(replace(wrld, replmap)) : rand(wrld)
  allstates = vcat(samples...)
  xs, ys = ntranspose(allstates)
  display(densityplot(xs, ys))
  m = migrationmovements(samples)
  return (m, samples, allstates)
end

terraincostsmat2 = [normal(LOW, 3.0) normal(NEUTRAL, 3.0) normal(HIGH, 3.0);
                    normal(LOW, 3.0) normal(HIGH, 3.0) normal(LOW, 3.0);
                    normal(NEUTRAL, 3.0) normal(NEUTRAL, 3.0) normal(NEUTRAL, 3.0)]

export runmodel, N, S, E, terraincostsmat, plotmigration
# What uncertainty are we doing inference over?
# Southeners have 

end