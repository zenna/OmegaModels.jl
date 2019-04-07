"Population Level Inverse Planning"
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
using ZenUtils
using DiscreteValueIteration
import Plots

include("gridworld.jl")

# TODO:
# - Show individual migration movements
# Make nicer version of the image
# Make a and b connected?
# Plot environment
# Ok we have plot prior, and we need matrix for tha 

# Coutnefactual a




"Get State Rewards as a Matrix"
function rewardmat(sgw)
  mat = Array{Float64}(undef, sgw.size...)
  for s  in POMDPs.states(sgw)
    x, y = s
    mat[x, y] = POMDPs.reward(sgw, s)
  end
  mat
end

function normat(x)
  mx = maximum(x)
  mn = minimum(x)
  (x .- mn) ./ (mx - mn)
end


"Piece of land (or water)"
struct Terrain
  id::Int
  pos::Vector{GWPos}
end

"Terrain with cost"
struct CostTerrain
  obj::Terrain
  cost::Float64
end

# Islands South, Norht, East #
S = Terrain(1, [GWPos(2,1), GWPos(3, 1), GWPos(4, 1), GWPos(3,2)])
N = Terrain(2, [GWPos(3,4), GWPos(2, 5), GWPos(3, 5), GWPos(4,5), GWPos(3,6)])
E = Terrain(3, [GWPos(7,2), GWPos(6, 3), GWPos(7,3), GWPos(7,4)])
islands = (S, N, E)

# wall = Terrain(3, [GWPos(1,3), GWPos(2,3), GWPos(3,3), GWPos(4,3), GWPos(5,3)])
wall = Terrain(4, [GWPos(1,3), GWPos(2,3), GWPos(3,3), GWPos(4,3), GWPos(5,3),
                   GWPos(1,4), GWPos(1, 5), GWPos(1, 6),
                   GWPos(5,4), GWPos(5, 5), GWPos(5, 6)])

watercost = -10.0
wallcost = -30

const invmap = Dict()
for ter in [S, N, E, wall]
  for pos in ter.pos
    invmap[pos] = ter
  end
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

# Can you have two different terrains
# if water is really hard to cross
# If u put wall
# wall is really hard to cross
# 

"Rewards Dict from costed items"
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

# unif(ω, vec) = vec[Int(ceil(rand(ω) ))]
# birthplace(ω, birthisland) = unif(ω, birthisland)

HIGH =  10.0
NEUTRAL = 0.0
LOW = -10.0


# terraincosts = Dict{Tuple{Terrain, Terrain}, RandVar}()
# terraincosts = Dict()
# # Dict[a, b] is belief (RandVar) of what someone from a thinks of cost of b

# # Northerners think The North is amazing and everywhere else horrible
# terraincosts[(N, N)] = normal(HIGH, 3.0)
# terraincosts[(N, E)] = normal(LOW, 3.0)
# terraincosts[(N, S)] = normal(LOW, 3.0)

# # Easterners are neutral
# terraincosts[(E, N)] = normal(NEUTRAL, 3.0)
# terraincosts[(E, E)] = normal(NEUTRAL, 3.0)
# terraincosts[(E, S)] = normal(NEUTRAL, 3.0)

# # Southeners have mixed opinions
# terraincosts[(S, N)] = normal(NEUTRAL, 3.0)
# terraincosts[(S, E)] = normal(HIGH, 3.0)
# terraincosts[(S, S)] = normal(LOW, 3.0)



terraincostsmat = [normal(LOW, 3.0) normal(NEUTRAL, 3.0) normal(HIGH, 3.0);
                   normal(LOW, 3.0) normal(HIGH, 3.0) normal(LOW, 3.0);
                   normal(NEUTRAL, 3.0) normal(NEUTRAL, 3.0) normal(NEUTRAL, 3.0)]

terraincostsmatrv = constant(terraincostsmat)
addwall = constant(false)

function solveworld(rng, ω; size = (7,6), defrward = watercost)
  birthisland = rand(rng, [islands...]) #FIXME? Should this be rng?
  birthplace_ = rand(rng, birthisland.pos)
  terraincostsmat_ = terraincostsmatrv(ω)
  objects = [CostTerrain(obj, terraincostsmat_[birthisland.id, obj.id](ω)) for obj in islands]
  # @show birthisland.id

  if addwall(ω)
    # @show "hi"
    push!(objects, CostTerrain(wall, wallcost))
  end
  # objects = [CostTerrain(obj, terraincosts[(birthisland, obj)](ω)) for obj in islands]
  rewards_ = rewards(objects)
  for (k, v) in rewards_
    rewards_[k] += randn(rng)
  end
  # display(rewards_)
  mdp = SimpleGridWorld(size = size,
                        rewards = rewards_,
                        defreward = defrward,
                        initialstate = birthplace_,
                        tprob = 0.99,
                        terminate_from = Set{GWPos}())
  # @grab mdp
  # @assert false
  # mdp = solve(mdp)
  # solver = MCTSSolver(n_iterations=1000, depth=20, exploration_constant=5.0, rng = rng)
  # policy = solve(solver, mdp)

  solver = ValueIterationSolver(max_iterations=10, belres=1e-6) # initializes the Solver type
  policy = solve(solver, mdp) # runs value iterations

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


function world(ω, npeople = 100, seed = 12345)
  rng  = Random.MersenneTwister(seed)
  statesseqs = Vector{GWPos}[]
  for i = 1:npeople
    push!(statesseqs, solveworld(rng, ω))
  end
  statesseqs
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

function plotmigration(amap, save = true)
  plt = Plots.heatmap(["S", "N", "E", "B", "W"], ["S", "N", "E"], amap,
                aspect_ratio = 1, label = false,
                colorbar = false, color = :grays,
                tickfontsize = 32)
end

function plotfinallevels(amap, save = true)
  plt = Plots.heatmap(amap',
                aspect_ratio = 1, label = false,
                colorbar = false, color = :grays,
                tickfontsize = 32)
end


function makeheatmap(allstates)
  mat = zeros(7,6)
  for (x,y) in allstates
    mat[x,y] += 1
  end
  mat
end

function makeplots(; n = 4, path)
  for i = 1:n
    (m, allseqs, allstates) = runmodel()
    plt = plotmigration(m)
    Plots.savefig(plt, joinpath(path, "prior$i.pdf"))

    plt2 = plotfinallevels(makeheatmap(allstates))
    Plots.savefig(plt2, joinpath(path, "priorfinal$i.pdf"))
  end
  for i = 1:n
    (m, allseqs, allstates) = runmodel(; usereplmap = true, replmap =  Dict(addwall => true))
    plt = plotmigration(m)
    Plots.savefig(plt, joinpath(path, "withwall$i.pdf"))

    plt2 = plotfinallevels(makeheatmap(allstates))
    Plots.savefig(plt2, joinpath(path, "withwallfinal$i.pdf"))
  end
  for i = 1:n
    (m, allseqs, allstates) =  runmodel(; usereplmap = true, replmap = Dict(addwall => true, 
                                      terraincostsmatrv => constant(terraincostsmat2)))

    plt = plotmigration(m)
    Plots.savefig(plt, joinpath(path, "cf$i.pdf"))

    plt2 = plotfinallevels(makeheatmap(allstates))
    Plots.savefig(plt2, joinpath(path, "cffinal$i.pdf"))
  end
end

export runmodel, N, S, E, terraincostsmat, plotmigration
# What uncertainty are we doing inference over?
# Southeners have 

end