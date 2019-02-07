"Causal inference: Is location of object A reason you can't see object B?"
module CanSee

import RayTrace
using RayTrace: Ray, Scene, render, Geometry, sceneintersect, FancySphere
using Omega
using Test

"Does `r` hit `scene` (first)?"
function canseetrc(r::Ray, scene::Scene, obj::Geometry)
  hit, geom, tnear = sceneintersect(r, scene)
  [hit && (geom == obj)]
end

"Can you see `obj` in `scene` (from origin)"
function cansee(scene::Scene, obj::Geometry; width = 100, height = 100, fov = 30.0)
  # @assert false
  img = render(scene; width = width, height = height, fov = fov,
               trc = (r, s) -> canseetrc(r, s, obj),
               image = zeros(Bool, width, height))
  any(img)
end

"Scene with LightSome example spheres which should create actual image"
function example_spheres()
  scene = [FancySphere(Float64[0.0,     20.0, -30],  3.0, Float64[0.00, 0.00, 0.00], 0.0, 0.0, Float64[3.0, 3.0, 3.0])]
  RayTrace.ListScene(scene)
end

function test()
  light = FancySphere(Float64[0.0,     20.0, -30],  3.0, Float64[0.00, 0.00, 0.00], 0.0, 0.0, Float64[3.0, 3.0, 3.0])
  obja = FancySphere(Float64[5.0,     -1, -15],     2.0, Float64[0.90, 0.76, 0.46], 1.0, 0.0, Float64[0.0, 0.0, 0.0])
  # Same as obja but in smaller location
  objb = FancySphere(Float64[5.0,     -1, -15],     1.0, Float64[0.90, 0.76, 0.46], 1.0, 0.0, Float64[0.0, 0.0, 0.0])

  scene = RayTrace.ListScene([light, obja, objb])
  @test cansee(scene, obja)
  @test !cansee(scene, objb)
end

end