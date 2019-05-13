module Interactive
using Makie
using AbstractPlotting
using GeometryTypes
using RayTrace

"Interactively render"
function interactiverender(scene3d; width = 100, height = 100, renderkwargs...)
  scene = Scene(resolution = (width, height))
  rorig = Point3(0.0)
  
  imgnode = Node(rand(width, height))
  image!(scene, imgnode, scale_plot = false)
  on(scene.events.keyboardbuttons) do button
    @show button
    if Keyboard.up in button rorig += Point(0.1, 0.0, 0.0) end
    if Keyboard.down in button rorig -= Point(0.1, 0.0, 0.0) end
    if Keyboard.left in button rorig += Point(0.0, 0.1, 0.0) end
    if Keyboard.right in button rorig -= Point(0.0, 0.1, 0.0) end
    if Keyboard.q in button rorig += Point(0.0, 0.0, 0.1) end
    if Keyboard.e in button rorig -= Point(0.0, 0.0, 0.1) end
    img = (RayTrace.renderfunc(scene3d; width = width, height = height, rorig = rorig, renderkwargs...))
    # @show typeof(img)
    push!(imgnode, img)
  end
  RecordEvents(scene, "output")
end


end