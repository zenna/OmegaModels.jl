

"Visualize a scene"
function vizscene(series, ball_pos, ball_radius)
  x, y = ntranspose(series)
  plt = plot(x, y, aspect_ratio = 1.0)
  # plot!(plt, Circle)
  pts = Plots.partialcircle(0, 2π, 100, ball_radius)
  x, y = Plots.unzip(pts)
  x = x .+ ball_pos[1]
  y = y .+ ball_pos[2]
  pts = collect(zip(x, y))
  plot!(Shape(x, y), c=:yellow, legend = false)
  # Add the ball
end