module Viz

using UnicodePlots

export unicodeplotmat

function samplemat(mat; maxsamples = 100, width = 50, height = 50)
  # For each element of matrix, add n samples in x y
  # at that position where 
  xs = Float64[]
  ys = Float64[]
  min_ = minimum(mat)
  max_ = maximum(mat)
  rng = max_ - min_
  for idx in CartesianIndices(mat)
    val = mat[idx]
    if max_ > min_
      intensity = (val - min_) / rng
      @assert 0.0 <= intensity <= 1.0 "$val / $rng"
      for i = 1:(intensity * maxsamples)
        push!(xs, idx[2])
        push!(ys, idx[1])
      end
    else
      push!(xs, idx[2] + rand())
      push!(ys, idx[1] + rand())
    end
  end
  (xs = xs, ys = ys)
end


"Plot a matrix using UnicodePlots"
function unicodeplotmat(mat; width = 120, height = 60, maxsamples = 60)
  xs, ys = samplemat(mat; maxsamples = maxsamples)
  if !isempty(xs)
    display(UnicodePlots.densityplot(xs, ys; width = width, height = height))
  end
end

end