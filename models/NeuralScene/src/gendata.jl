"Generate Synthetic Data"
module GenData
using RayTrace
using RayTrace: Ray, Scene, sceneintersect, trcdepth
using Omega
using GeometryTypes

export ex_data, gendata, datarv

function datarv(ω, scene, render_params)
  rorig = Vec3(uniform(ω, 0, 1), uniform(ω, 0, 1), uniform(ω, 0, 1))
  img = RayTrace.renderfunc(scene; rorig = rorig, render_params...)
  (rorig = rorig, img = img)
end

"Returns an Omega random variable"
function gendata(; scene = RayTrace.example_spheres(),
                   render_params = (width = 100, height = 100, fov = 30.0, trc = trcdepth))
  # function datarv(ω)
  #   rorig = Vec3(uniform(ω, 0, 1), uniform(ω, 0, 1), uniform(ω, 0, 1))
  #   img = RayTrace.renderfunc(scene; rorig = rorig, render_params...)
  #   (rorig = rorig, img = img)
  # end
  ciid(datarv, scene, render_params)
end

function ex_data()
  scene = RayTrace.example_spheres()
  # zt: need to include camera pos in arguments
  render_params = (width = 100, height = 100, fov = 30.0, trc = trcdepth)
  img = RayTrace.renderfunc(scene; render_params...)
  (scene = scene, render_params = render_params, img = img)
end

end