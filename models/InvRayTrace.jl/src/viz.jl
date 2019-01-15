using MeshCat
using GeometryTypes
using CoordinateTransformations
import RayTrace
import ColorTypes: RGBA, RGB

function visualize(sphere::RayTrace.FancySphere, name, vis)
  s = HyperSphere(Point3f0(sphere.center), Float32(sphere.radius))
  material = MeshPhongMaterial(color=@show RGBA([sphere.surface_color; 1.0]...))
  setobject!(vis[name], s, material)
end

"Visualize a scene using MeshCat"
function visualize(scene::RayTrace.Scene; vis = Visualizer())
  for (i, geom) in enumerate(scene.geoms)
    # @show "hello"
    visualize(geom, "obj$i", vis)
  end
  vis
end