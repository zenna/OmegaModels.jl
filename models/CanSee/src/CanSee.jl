"Causal inference: Is location of object A reason you can't see object B?"
module CanSee

import RayTrace
using RayTrace: Ray, Scene, render, sceneintersect, msphere
using Omega
using Test
using GeometryTypes
using Omega.Soft: DualSoftBool, SoftBool

function Omega.softeq(a::RayTrace.US, b::RayTrace.US)
  res = (a.r ==ₛ b.r) & (a.center ==ₛ b.center)
end

"Does `r` hit `obj`  in `scene` first?"
function canseetrc(r::Ray, scene::Scene, obj)
  hit, geom, tnear = sceneintersect(r, scene)
  hit && (geom == obj)
end

"Does `r` hit `obj`  in `scene` first?"
function canseetrcₛ(r::Ray, scene::Scene, obj)
  hit, geom, tnear = sceneintersect(r, scene)
  geom ==ₛ obj # FIXME: This is wrong because if theres no intersection itll jsut return some arbitrary thing
end

"Can you see `obj` in `scene` (from origin)"
function cansee(scene::Scene, obj; width = 100, height = 100, fov = 30.0)
  # @assert false
  img = render(scene; width = width, height = height, fov = fov,
               trc = (r, s) -> canseetrcₛ(r, s, obj),
               image = Array{DualSoftBool{SoftBool{Float64}}}(undef, width, height))
  anyₛ(img)
end

"Scene with LightSome example spheres which should create actual image"
function mixed_scene()
  scene = [msphere(Point(0.0, -10004, -20), 10000.0, Vec3(0.20, 0.20, 0.20), 0.0, 0.0, Vec3(0.0, 0.0, 0.0)),
           msphere(Point(0.0, 0.0, -20), 4.0, Vec3(1.0, 0.32, 0.36), 1.0, 0.5, zeros(Vec3)),
           msphere(Point(0.0, 0.0, -25), 2.0, Vec3(0.0, 0.32, 1.0), 1.0, 0.5, zeros(Vec3)),
           msphere(Point(0.0, 20.0, -30), 3.0, zeros(Vec3), 0.0, 0.0, Vec3(3.0, 3.0, 3.0))]
RayTrace.ListScene(scene)
end


function test()
  light = msphere(Point(0.0, 20.0, -30), 3.0, zeros(Vec3), 0.0, 0.0, Vec3(3.0, 3.0, 3.0))
  base = msphere(Point(0.0, -10004, -20), 10000.0, Vec3(0.20, 0.20, 0.20), 0.0, 0.0, Vec3(0.0, 0.0, 0.0))
  obja = msphere(Point(0.0, 0.0, -20), 4.0, Vec3(1.0, 0.32, 0.36), 1.0, 0.5, zeros(Vec3))
  objb = msphere(Point(0.0, 0.0, -25), 2.0, Vec3(0.0, 0.32, 1.0), 1.0, 0.5, zeros(Vec3))
  scene = RayTrace.ListScene([light, base, obja, objb])
  @test Bool(cansee(scene, obja))
  @test Bool(!cansee(scene, objb))
  
  # Actual Causality
  objapos = constant(Point(0.0, 0.0, -20))
  obja = lift(msphere)(objapos, 4.0, Vec3(1.0, 0.32, 0.36), 1.0, 0.5, zeros(Vec3))
  scene = ciid(ω -> RayTrace.ListScene([light, base, obja(ω), objb]))
  cantseeb = ciid(ω -> !cansee(scene(ω), objb))
  ω = defΩ()()
 
  iscausebf(ω, objapos ==ₛ objapos(ω), cantseeb, [objapos]; sizes = [3], proj = v -> Point(v[1], v[2], v[3]))

end


end