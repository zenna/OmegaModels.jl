"Generate Synthetic Data"
module GenData
using RayTrace
using RayTrace: Ray, Scene, sceneintersect, trcdepth

export ex_data


function gendata(scene, )
end

function ex_data()
  scene = RayTrace.example_spheres()
  # zt: need to include camera pos in arguments
  render_params = (width = 3, height = 3, fov = 30.0, trc = trcdepth)
  img = RayTrace.renderfunc(scene; render_params...)
  (scene = scene, render_params = render_params, img = img)
end

end