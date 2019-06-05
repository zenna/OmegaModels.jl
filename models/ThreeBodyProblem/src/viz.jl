using MeshCat
using GeometryTypes
import ColorTypes: RGBA, RGB
using CoordinateTransformations

viz(body; vis) = setobject!(vis, body)

function visualize(body, vis)
  s = HyperSphere(Point3f0(body.pos), Float32(body.r))
  material = MeshPhongMaterial(color = RGBA([body.color; 1.0]...))
  setobject!(vis[body.name], s, material)
end


"Vizualise the tiem series"
function viz(bodies; vis = Visualizer())
  for body in bodies
    visualize(body, vis)
  end
  vis
end

"Animate the time series"
function animate(bodiesstream; vis = viz(bodiesstream[1]), anim = Animation())
  nframes = length(bodiesstream)
  for i = 2:nframes
    atframe(anim, vis, i) do frame
        bodies = bodiesstream[i]
        for body in bodies
          settransform!(frame[body.name], Translation(body.pos))
        end
    end
  end
  setanimation!(vis, anim)
  open(vis)
  (vis = vis, anim = anim)
end