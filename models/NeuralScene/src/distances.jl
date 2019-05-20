module Distances
using LinearAlgebra
export distance, dist_ignorebg

"Euclidiean distance"
distance(x, y) = norm(x - y)

"Distance ignoring background distance"
function dist_ignorebg(x, target; background = 100.0)
  total = 0.0
  i = 1
  for idx in CartesianIndices(x)
    if target[idx] != background
      i += 1
      total += (target[idx] - x[idx])^2
    end
  end
  @show i
  sqrt(total)
end

end