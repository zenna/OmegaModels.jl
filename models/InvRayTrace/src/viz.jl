using MeshCat
using GeometryTypes
using CoordinateTransformations
import RayTrace
import ColorTypes: RGBA, RGB
import Makie
import GeometryTypes
using LinearAlgebra: norm

## MeshCat
## =======
function visualize(sphere::RayTrace.MaterialGeom, name, vis)
  s = HyperSphere(Point3f0(sphere.center), Float32(sphere.r))
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

## Wireframe
## =========
"Convert FancySphere to GeometryTypes Sphere "
gtype(sphere::RayTrace.Sphere) =
  GLNormalUVMesh(GeometryTypes.HyperSphere(Point3f0(sphere.center), Float32(sphere.r)))
  
gtype(s::RayTrace.Scene) = Makie.merge([gtype(geom) for geom in s.geoms][1:end-2])

## Intersections
## =============
"Is the point in the sphere"
inobj(sphere::RayTrace.Sphere, x, y, z) = norm(sphere.center - [x, y, z]) < sphere.r
inobj(geoms, x, y, z) = sum([inobj(geom, x, y, z) for geom in geoms])

mins(sphere::RayTrace.Sphere) = min.(sphere.center .- sphere.r, sphere.center .+ sphere.r)
maxs(sphere::RayTrace.Sphere) = max.(sphere.center .- sphere.r, sphere.center .+ sphere.r)
function mins(geoms)
  mins_ = [Inf, Inf, Inf]
  for geom in geoms
    mins_ = min.(mins_, mins(geom))
  end
  mins_
end

function maxs(geoms)
  maxs_ = [-Inf, -Inf, -Inf]
  for geom in geoms
    maxs_ = max.(maxs_, maxs(geom))
  end
  maxs_
end

"Voxel grid showing intersection between objects"
function intersectvoxels(geoms; step = 0.1, mins_ =  mins(geoms), maxs_ = maxs(geoms))
  xs, ys, zs = ((lb, ub) -> lb:step:ub).(mins_, maxs_)
  mat = Array{Float32}(undef, length(xs), length(ys), length(zs))
  for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys), (iz, z) in enumerate(zs)
    mat[ix, iy, iz] = inobj(geoms, x, y, z)
  end
  mat
end

"Vizualise the voxel grid"
vizintersectvoxels(mat; max_intersects = maximum(mat), algorithm = :mip, kwargs...) =
  Makie.volume(mat ./ max_intersects; algorithm = algorithm, kwargs...)


"Update camera to orthograpic projection"
function orthographic!(scene)
  cam = Makie.cameracontrols(scene)
  dir = scene.limits[].widths ./ 2.
  dir_scaled = Vec3f0(
      dir[1] * scene.transformation.scale[][1],
      0.0,
      dir[3] * scene.transformation.scale[][2],
  )
  cam.upvector[] = (0.0, 0.0, 1.0)
  cam.lookat[] = scene.limits[].origin + dir_scaled
  cam.eyeposition[] = (cam.lookat[][1], cam.lookat[][2] + 6.3, cam.lookat[][3])
  cam.projectiontype[] = AbstractPlotting.Orthographic
  update_cam!(scene, cam)
  # stop scene display from centering, which would overwrite the camera paramter we just set
  scene.center = false
  scene
end