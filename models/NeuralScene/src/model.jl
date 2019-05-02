module Model

import RayTrace
using ..Linearize: linearize, unlinearize
import ..Linearize
using DSLearn: net
using RayTrace: Ray, Scene, sceneintersect

export DeepScene

"Neural representation of a scene"
struct DeepScene{T} <: RayTrace.Scene
  ir::T
end

function Linearize.unlinearize(::Type{NamedTuple{(:hit, :geom, :tnear), Tuple{Bool,Float64,Float64}}}, out)
  NamedTuple{(:hit, :geom, :tnear)}((true, false, first(out)))
end

function RayTrace.sceneintersect(r::Ray, scene::DeepScene)
  inp = linearize(Vector{Float64}, (r, scene)) # zt: what type should this be?
  net_ = net(typeof(sceneintersect), Ray, DeepScene)
  out = net_(inp)
  unlinearize(NamedTuple{(:hit, :geom, :tnear), Tuple{Bool,Float64,Float64}}, out)
end
  # hit, geom, tnear # zt: how to delinearize
  # Hit as a boolean, geom as the object it hit, and tnear is the distance to
  # closest poitn

  # Problem 1. Geoms
  # DeepScene doesn't have explicit objects
  # So either we 1. convert to an object, using a neural network
  # i.e. either a particular structure, e.g. a sphere
  # Or a NeuralGeom
  # Or i refactor the code to not need a geom

  # geom is used for transparency, reflection, surface_color, emission_color, surface_color
  # Possible todos:
  # 1. create nets for all of these interfaces
  # 2. Have another net for `light`
  # 3. Make simpler renderer

  # For now just try depth trace
  # Problem 2. dithit!
  # The issue is that if it did not hit then tnear is unimportant
  # Do we want to cast them?
# end

end