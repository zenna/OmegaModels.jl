
"Four points (x, y) - corners of `box`"
function corners(box)
  ((box.x, box.y),
   (box.x + box.Δx, box.y),
   (box.x + box.Δx, box.y - box.Δy),
   (box.x, box.y -  box.Δy))
end

"Draw Box"
function draw(obj, canvas, color = :blue)
  corners_ = corners(obj)
  for i = 1:length(corners_)
    p1 = corners_[i]
    p2 = i < length(corners_) ? corners_[i + 1] : corners_[1]
    "Fix aspect ratio (account that uncicode is taller than wide)"
    fixao(x, y; aspectratio = 0.5) = (x, Int(y * aspectratio))
     
    lines!(canvas, p1..., p2..., color)
  end
  canvas
end

"Fix aspect ratio (account that uncicode is taller than wide)"
fixao(x, y; aspectratio = 0.5) = (x, Int(y * aspectratio))

"Draw Scene"
function draw(scene::Scene,
              canvas = BrailleCanvas(fixao(64, 32)..., origin_x = -50.0, origin_y = -50.0,
                                     width = scene.camera.Δx + 10, height = scene.camera.Δy + 10))
  draw(scene.camera, canvas, :red)
  foreach(obj -> draw(obj, canvas, :blue), scene.objects)
  canvas
end

"Draw a sequence of frames"
function viz(vid, sleeptime = 0.02)
  foreach(vid) do o
    display(draw(o))
    sleep(sleeptime)
  end
end