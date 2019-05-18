module Viz

using UnicodePlots

export unicodeplotmat

function samplemat(mat; multiply = 10, width = 50, height = 50)
  # For each element of matrix, add n samples in x y
  # at that position where 
  xs = Float64[]
  ys = Float64[]
  min_ = minimum(mat)
  max_ = maximum(mat)
  rng = max_ - min_
  for idx in CartesianIndices(mat)
    val = mat[idx]
    
    for i = 1:(val / rng * multiply)
      push!(xs, idx[2])
      push!(ys, idx[1])
    end
  end
  (xs = xs, ys = ys)
end


"Plot a matrix using UnicodePlots"
function unicodeplotmat(mat; width = 60, height = 30)
  xs, ys = samplemat(mat; multiply = 1000)
  if !isempty(xs)
    display(UnicodePlots.densityplot(xs, ys; width = width, height = height))
  end
end

end