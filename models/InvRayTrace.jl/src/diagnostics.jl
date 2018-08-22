# Diagnostics

Δ(a::Sphere, b::Sphere) = norm(a.center - b.center) + abs(a.radius - b.radius)
Δ(a::Scene, b::Scene) = surjection(a.geoms, b.geoms)

"distance betwee two scenes"
function hausdorff(s1, s2, Δ = Δ)
  Δm(x, S) = minimum([Δ(x, y) for y in S])
  max(maximum([Δm(e, s2) for e in s1]), maximum([Δm(e, s1) for e in s2]))
end

function plothist(truth, samples, plt = plot())
  distances = Δ.(truth, samples)
  histogram(distances)
end